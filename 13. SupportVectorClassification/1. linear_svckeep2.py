# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.svm import LinearSVC
from scipy.stats import uniform
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt


from sklearn.model_selection import cross_validate, \
  RandomizedSearchCV, StratifiedKFold
import sklearn.metrics as skmet

import os
import sys
sys.path.append(os.getcwd() + "/helperfunctions")
from preprocfunc import OutlierTrans

pd.set_option('display.width', 120)
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.0f}'.format

# setup the features and target
nbagames = pd.read_csv("data/nbagames2017plus.csv", parse_dates=['GAME_DATE'])
nbagames = nbagames.loc[nbagames.WL_HOME.isin(['W','L'])]
nbagames.shape
#nbagames.iloc[0:2,0:21].T

nbagames['WL_HOME'] = \
  np.where(nbagames.WL_HOME=='L',0,1).astype('int')
  
nbagames.WL_HOME.value_counts(dropna=False)

# take a look at some of the data

# identify numeric and categorical data
num_cols = ['FGM_HOME','FGA_HOME','FG_PCT_HOME',
  'FG3M_HOME',  'FG3A_HOME','FG3_PCT_HOME','FTM_HOME',
  'FTA_HOME','FT_PCT_HOME',  'OREB_HOME','DREB_HOME',
  'REB_HOME','AST_HOME','STL_HOME','BLK_HOME','TOV_HOME',
  'FGM_AWAY','FGA_AWAY','FG_PCT_AWAY','FG3M_AWAY',
  'FG3A_AWAY','FG3_PCT_AWAY','FTM_AWAY','FTA_AWAY',
  'FT_PCT_AWAY','OREB_AWAY','DREB_AWAY','REB_AWAY',
  'AST_AWAY','STL_AWAY','BLK_AWAY','TOV_AWAY',
  'PTS_PAINT_HOME','PTS_2ND_CHANCE_HOME',
  'PTS_PAINT_AWAY',  'PTS_2ND_CHANCE_AWAY',
  'PTS_OFF_TO_HOME','PTS_OFF_TO_AWAY']
cat_cols = ['TEAM_ABBREVIATION_HOME','SEASON']

# create training and testing DataFrames

nbagames[['WL_HOME'] + num_cols].agg(['count','min','median','max']).T

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



# fit an SVR model
svc = LinearSVC(max_iter=1000000, random_state=0)

rfe = RFE(estimator=svc, n_features_to_select=15)

#pipe1 = make_pipeline(coltrans, SelectKBest(score_func=chi2, k=15), svc)
pipe1 = make_pipeline(coltrans, rfe, svc)

pipe1.fit(X_train, y_train.values.ravel())

new_cat_cols = \
  pipe1.named_steps['columntransformer'].\
  named_transformers_['cat'].\
  named_steps['onehotencoder'].\
  get_feature_names(cat_cols)


new_cols = np.concatenate((new_cat_cols, np.array(num_cols)))
sel_cols = new_cols[pipe1['rfe'].get_support()]
sel_cols

coefs = pipe1['linearsvc'].coef_
coefwithlabs = np.column_stack((coefs.ravel(), sel_cols))
np.sort(coefwithlabs, axis=0)[::-1]

pd.Series(abs(pipe1['linearsvc'].coef_[0]), index=sel_cols).nlargest(10).plot(kind='barh')
pd.Series(abs(pipe1['linearsvc'].coef_[0]), index=sel_cols).plot(kind='barh')

pred = pipe1.predict(X_test)

print("accuracy: %.2f, sensitivity: %.2f, specificity: %.2f, precision: %.2f"  %
  (skmet.accuracy_score(y_test.values.ravel(), pred),
  skmet.recall_score(y_test.values.ravel(), pred),
  skmet.recall_score(y_test.values.ravel(), pred, pos_label=0),
  skmet.precision_score(y_test.values.ravel(), pred)))


# do kfold cross validation
kf = StratifiedKFold(n_splits=7, shuffle=True, random_state=0)

scores = cross_validate(pipe1, X_train, y_train.values.ravel(), \
  scoring=['accuracy','precision','recall','f1'], cv=kf, n_jobs=-1)


print("accuracy: %.2f, precision: %.2f, sensitivity: %.2f, f1: %.2f"  %
  (np.mean(scores['test_accuracy']),\
  np.mean(scores['test_precision']),\
  np.mean(scores['test_recall']),\
  np.mean(scores['test_f1'])))

# do a grid search to find the best value of alpha
#svc = LinearSVC(max_iter=10000, class_weight='balanced', random_state=0)

svc_params = {
 'rfe__n_features_to_select': np.arange(1, len(new_cols)),
 'linearsvc__C': uniform(loc=0, scale=100)
}

rs = RandomizedSearchCV(pipe1, svc_params, cv=7, scoring='roc_auc')
rs.fit(X_train, y_train.values.ravel())

rs.best_params_



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


