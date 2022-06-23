# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform

import sklearn.metrics as skmet

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

# construct a pipeline with preprocessing, feature selection, and logistic model

#kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)

lr = LogisticRegression(random_state=0, class_weight='balanced',
  solver='liblinear', penalty="l1", max_iter=1000)

pipe1 = make_pipeline(coltrans, lr)

reg_params = {
 'logisticregression__C': uniform(loc=0, scale=10)
}

uniform(loc=0, scale=10)

rs = RandomizedSearchCV(pipe1, reg_params, cv=5, scoring='roc_auc')
rs.fit(X_train, y_train.values.ravel())

rs.best_params_
rs.best_score_


# get predictions and residuals
pred = rs.predict(X_test)

cm = skmet.confusion_matrix(y_test, pred)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
cmplot.plot()
cmplot.ax_.set(title='Heart Disease Prediction Confusion Matrix', 
  xlabel='Predicted Value', ylabel='Actual Value')


# another way to calculate the metrics
accuracy, sensitivity, specificity, precision = \
  skmet.accuracy_score(y_test.values.ravel(), pred),\
  skmet.recall_score(y_test.values.ravel(), pred),\
  skmet.recall_score(y_test.values.ravel(), pred,  pos_label=0),\
  skmet.precision_score(y_test.values.ravel(), pred)
accuracy, sensitivity, specificity, precision

healthinfo.shape
h2 = healthinfo.sample(30000)
h2.shape
h2.to_csv("data/healthinfosample.csv")
