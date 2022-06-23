# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

#from sklearn.metrics import accuracy_score
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import seaborn as sb

import os
import sys
sys.path.append(os.getcwd() + "/helperfunctions")
from preprocfunc import OutlierTrans

pd.set_option('display.width', 78)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.3f}'.format

# load the health information data
healthinfo = pd.read_csv("data/healthinfo.csv")
healthinfo.head(7)
healthinfo.info()

healthinfo.diabetic.value_counts()
healthinfo.heartdisease.value_counts()

healthinfo['diabetic'] = \
  np.where(healthinfo.diabetic.str[0:2]=='No','No','Yes')
healthinfo['heartdisease'] = \
  np.where(healthinfo.heartdisease=='No',0,1).astype('int')

healthinfo.groupby(['heartdisease','stroke','diabetic','asthma','kidneydisease','skincancer']).size().reset_index()

# take a look at some of the data

# identify numeric and categorical data
num_cols = ['bmi','physicalhealthbaddays','mentalhealthbaddays',
  'sleeptimenightly']
binary_cols = ['smoking','alcoholdrinkingheavy','stroke',
  'walkingdifficult','diabetic','physicalactivity','asthma',
  'kidneydisease','skincancer']
cat_cols = ['gender','agecategory','ethnicity','genhealth']

# generate some counts and descriptive data
healthinfo[binary_cols].\
  apply(pd.value_counts, normalize=True).T

for col in healthinfo[cat_cols].columns:
  print(col, "----------------------",
  healthinfo[col].value_counts(normalize=True).sort_index(),
  sep="\n", end="\n\n")

healthinfo[num_cols].agg(['count','min','median','max']).T

# create training and testing DataFrames
X_train, X_test, y_train, y_test =  \
  train_test_split(healthinfo[num_cols + binary_cols + cat_cols],\
  healthinfo[['heartdisease']], test_size=0.2, random_state=0)


# setup column transformations
ohe = OneHotEncoder(drop='first', sparse=False)

standtrans = make_pipeline(OutlierTrans(3),SimpleImputer(strategy="median"),
  StandardScaler())
bintrans = make_pipeline(ohe)
cattrans = make_pipeline(ohe)
coltrans = ColumnTransformer(
  transformers=[
    ("stand", standtrans, num_cols),
    ("bin", bintrans, binary_cols),
    ("cat", cattrans, cat_cols),
  ]
)

# setup column transformations
ohe = OneHotEncoder(drop='first', sparse=False)

standtrans = make_pipeline(OutlierTrans(3),SimpleImputer(strategy="median"),
  StandardScaler())
cattrans = make_pipeline(ohe)
coltrans = ColumnTransformer(
  transformers=[
    ("stand", standtrans, num_cols),
    ("cat", cattrans, cat_cols),
  ]
)

# construct a pipeline with preprocessing, feature selection, and logistic model
lr = LogisticRegression(random_state=0, multi_class='multinomial',
  penalty='none', solver='sag', class_weight='balanced')

pipe1 = make_pipeline(coltrans, lr)

pipe1.fit(X_train, y_train.values.ravel())

# get predictions and residuals
pred = pipe1.predict(X_test)
pd.Series(pred).value_counts().sort_index()
y_test.value_counts()

cm = skmet.confusion_matrix(y_test, pred)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix=cm,
  display_labels=['None','Heat','Power','Overstrain','Other'])
cmplot.plot()
cmplot.ax_.set(title='Heart Disease Prediction Confusion Matrix', 
  xlabel='Predicted Value', ylabel='Actual Value')

print(skmet.classification_report(y_test, pred,
  target_names=['None','Heat','Power','Overstrain','Other']))


lr = LogisticRegression(random_state=0, multi_class='multinomial',
  penalty='none', solver='sag')

pipe1 = make_pipeline(coltrans, lr)


pipe1.fit(X_train, y_train.values.ravel())

pred2 = pipe1.predict(X_test)

cm = skmet.confusion_matrix(y_test, pred2)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix=cm,
  display_labels=['None','Heat','Power','Overstrain','Other'])
cmplot.plot()
cmplot.ax_.set(title='Heart Disease Prediction Confusion Matrix', 
  xlabel='Predicted Value', ylabel='Actual Value')

print(skmet.classification_report(y_test, pred2,
  target_names=['None','Heat','Power','Overstrain','Other']))
