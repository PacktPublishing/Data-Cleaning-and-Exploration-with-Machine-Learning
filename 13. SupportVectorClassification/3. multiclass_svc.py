# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
#from sklearn.multiclass import OneVsRestClassifier
from scipy.stats import uniform
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
machinefailuretype = pd.read_csv("data/machinefailuretype.csv")
machinefailuretype.info()
machinefailuretype.head()
machinefailuretype.failtype.\
  value_counts(dropna=False).sort_index()


def setcode(typetext):
  if (typetext=="No Failure"):
    typecode = 1
  elif (typetext=="Heat Dissipation Failure"):
    typecode = 2
  elif (typetext=="Power Failure"):
    typecode = 3
  elif (typetext=="Overstrain Failure"):
    typecode = 4
  else:
    typecode = 5
  return typecode

machinefailuretype["failtypecode"] = \
  machinefailuretype.apply(lambda x: setcode(x.failtype), axis=1)


# identify numeric and categorical data
num_cols = ['airtemp','processtemperature','rotationalspeed',
  'torque','toolwear']
cat_cols = ['machinetype']

machinefailuretype[num_cols].agg(['min','median','max']).T

# create training and testing DataFrames
X_train, X_test, y_train, y_test =  \
  train_test_split(machinefailuretype[num_cols + cat_cols],\
  machinefailuretype[['failtypecode']],\
  stratify=machinefailuretype[['failtypecode']], \
  test_size=0.2, random_state=0)

# setup column transformations
ohe = OneHotEncoder(drop='first', sparse=False)

cattrans = make_pipeline(ohe)
standtrans = make_pipeline(OutlierTrans(3),SimpleImputer(strategy="median"),
  MinMaxScaler())
coltrans = ColumnTransformer(
  transformers=[
    ("cat", cattrans, cat_cols),
    ("stand", standtrans, num_cols),
  ]
)


# add feature selection and a linear model to the pipeline and look at the parameter estimates
svc = SVC(class_weight='balanced', probability=True)

pipe1 = make_pipeline(coltrans, svc)

svc_params = [
  {
    'svc__kernel': ['rbf'],
    'svc__C': uniform(loc=0, scale=20),
    'svc__gamma': uniform(loc=0, scale=100),
    'svc__decision_function_shape': ['ovr','ovo']
  },
  {
    'svc__kernel': ['poly'],
    'svc__degree': np.arange(0,6),
    'svc__C': uniform(loc=0, scale=20),
    'svc__gamma': uniform(loc=0, scale=100),
    'svc__decision_function_shape': ['ovr','ovo']
  },
  {
    'svc__kernel': ['linear','sigmoid'],
    'svc__C': uniform(loc=0, scale=20),
    'svc__decision_function_shape': ['ovr','ovo']
  }
]

rs = RandomizedSearchCV(pipe1, svc_params, cv=7, scoring="roc_auc_ovr", n_iter=10)
rs.fit(X_train, y_train.values.ravel())

rs.best_params_

rs.best_score_
#scores = list(zip(rs.cv_results_['mean_test_score'], rs.cv_results_['params']))
#scores

results = \
  pd.DataFrame(rs.cv_results_['mean_test_score'], \
    columns=['meanscore']).\
  join(pd.json_normalize(rs.cv_results_['params'])).\
  sort_values(['meanscore'], ascending=False)

results


pred = rs.predict(X_test)

cm = skmet.confusion_matrix(y_test, pred)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix=cm,
   display_labels=['None', 'Heat','Power','Overstrain','Other'])
cmplot.plot()
cmplot.ax_.set(title='Machine Failure Type Confusion Matrix', 
  xlabel='Predicted Value', ylabel='Actual Value')

print(skmet.classification_report(y_test, pred,
  target_names=['None', 'Heat','Power','Overstrain','Other']))
