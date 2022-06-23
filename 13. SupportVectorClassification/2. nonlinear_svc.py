# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import RandomizedSearchCV
import sklearn.metrics as skmet
#import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.getcwd() + "/helperfunctions")
from preprocfunc import OutlierTrans

pd.set_option('display.width', 78)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.2f}'.format

# setup the features and target
nbagames = pd.read_csv("data/nbagames2017plus.csv", parse_dates=['GAME_DATE'])
nbagames = nbagames.loc[nbagames.WL_HOME.isin(['W','L'])]
nbagames['WL_HOME'] = \
  np.where(nbagames.WL_HOME=='L',0,1).astype('int')


# identify numeric and categorical data
num_cols = ['FG_PCT_HOME','FTA_HOME','FG3_PCT_HOME',
  'FTM_HOME','FT_PCT_HOME','OREB_HOME','DREB_HOME',
  'REB_HOME','AST_HOME','STL_HOME','BLK_HOME','TOV_HOME',
  'FG_PCT_AWAY','FTA_AWAY','FG3_PCT_AWAY',
  'FT_PCT_AWAY','OREB_AWAY','DREB_AWAY','REB_AWAY',
  'AST_AWAY','STL_AWAY','BLK_AWAY','TOV_AWAY']
cat_cols = ['TEAM_ABBREVIATION_HOME','SEASON']


X_train, X_test, y_train, y_test =  \
  train_test_split(nbagames[num_cols + cat_cols],\
  nbagames[['WL_HOME']], test_size=0.2, random_state=0)


# setup column transformations
ohe = OneHotEncoder(drop='first', sparse=False)

cattrans = make_pipeline(ohe)
standtrans = make_pipeline(OutlierTrans(2),
  SimpleImputer(strategy="median"), MinMaxScaler())
coltrans = ColumnTransformer(
  transformers=[
    ("cat", cattrans, cat_cols),
    ("stand", standtrans, num_cols)
  ]
)


# add feature selection and a linear model to the pipeline and look at the parameter estimates
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=18)

svc = SVC()

pipe1 = make_pipeline(coltrans, rfe, svc)

svc_params = [
  {
    'svc__kernel': ['rbf'],
    'svc__C': uniform(loc=0, scale=20),
    'svc__gamma': uniform(loc=0, scale=100)
  },
  {
    'svc__kernel': ['poly'],
    'svc__degree': np.arange(0,6),
    'svc__C': uniform(loc=0, scale=20),
    'svc__gamma': uniform(loc=0, scale=100)
  },
  {
    'svc__kernel': ['linear','sigmoid'],
    'svc__C': uniform(loc=0, scale=20)
  }
]

rs = RandomizedSearchCV(pipe1, svc_params, cv=7, scoring="accuracy")
rs.fit(X_train, y_train.values.ravel())

rs.best_params_
rs.best_score_

list(zip(rs.cv_results_['mean_test_score'], rs.cv_results_['params']))
rs.cv_results_['params']

results = \
  pd.DataFrame(rs.cv_results_['mean_test_score'], \
    columns=['meanscore']).\
  join(pd.json_normalize(rs.cv_results_['params'])).\
  sort_values(['meanscore'], ascending=False)

results



pred = rs.predict(X_test)

print("accuracy: %.2f, sensitivity: %.2f, specificity: %.2f, precision: %.2f"  %
  (skmet.accuracy_score(y_test.values.ravel(), pred),
  skmet.recall_score(y_test.values.ravel(), pred),
  skmet.recall_score(y_test.values.ravel(), pred, pos_label=0),
  skmet.precision_score(y_test.values.ravel(), pred)))

cm = skmet.confusion_matrix(y_test, pred)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
cmplot.plot()
cmplot.ax_.set(title='Heart Disease Prediction Confusion Matrix', 
  xlabel='Predicted Value', ylabel='Actual Value')

