# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.svm import SVC
from scipy.stats import uniform
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTENC
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
healthinfo = pd.read_csv("data/healthinfosmallsample.csv")

healthinfo['diabetic'] = \
  np.where(healthinfo.diabetic.str[0:2]=='No','No','Yes')
healthinfo['heartdisease'] = \
  np.where(healthinfo.heartdisease=='No',0,1).astype('int')
  
# take a look at some of the data

# identify numeric and categorical data
num_cols = ['bmi','physicalhealthbaddays','mentalhealthbaddays',
  'sleeptimenightly']
binary_cols = ['smoking','alcoholdrinkingheavy','stroke',
  'walkingdifficult','diabetic','physicalactivity','asthma',
  'kidneydisease','skincancer']
cat_cols = ['gender','agecategory','ethnicity','genhealth']

# create training and testing DataFrames
X_train, X_test, y_train, y_test =  \
  train_test_split(healthinfo[num_cols + binary_cols + cat_cols],\
  healthinfo[['heartdisease']], test_size=0.2, random_state=0)


# setup column transformations
ohe = OneHotEncoder(drop='first', sparse=False)

bintrans = make_pipeline(ohe)
cattrans = make_pipeline(ohe)
standtrans = make_pipeline(OutlierTrans(2),
  SimpleImputer(strategy="median"),StandardScaler())
coltrans = ColumnTransformer(
  transformers=[
    ("bin", bintrans, binary_cols),
    ("cat", cattrans, cat_cols),
    ("stand", standtrans, num_cols),
  ]
)

coltrans.fit(X_train.sample(1000))

new_binary_cols = \
  coltrans.\
  named_transformers_['bin'].\
  named_steps['onehotencoder'].\
  get_feature_names(binary_cols)
new_cat_cols = \
  coltrans.\
  named_transformers_['cat'].\
  named_steps['onehotencoder'].\
  get_feature_names(cat_cols)

new_cols = np.concatenate((new_binary_cols, new_cat_cols, np.array(num_cols)))


# add feature selection and a linear model to the pipeline and look at the parameter estimates
svc = SVC(kernel='rbf', class_weight='balanced')

catcolscnt = new_binary_cols.shape[0] + new_cat_cols.shape[0]
smotenc = SMOTENC(categorical_features=np.arange(0,catcolscnt), random_state=0)

#pipe1 = make_pipeline(coltrans, smotenc, svc)
pipe1 = make_pipeline(coltrans, svc)

svc_params = [
  {
    'svc__kernel': ['rbf','poly'],
    'svc__C': uniform(loc=0, scale=20),
    'svc__gamma': uniform(loc=0, scale=100)
  },
  {
    'svc__kernel': ['linear','sigmoid'],
    'svc__C': uniform(loc=0, scale=20)
  }
]

rs = RandomizedSearchCV(pipe1, svc_params, cv=7, scoring="roc_auc")
rs.fit(X_train, y_train.values.ravel())

rs.best_params_
rs.best_score_

rs.cv_results_

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

print(skmet.classification_report(y_test, pred))
