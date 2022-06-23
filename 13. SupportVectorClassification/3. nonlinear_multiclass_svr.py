# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.svm import SVC
from scipy.stats import uniform
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import RandomizedSearchCV
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, \
  RandomizedSearchCV, StratifiedKFold

import os
import sys
sys.path.append(os.getcwd() + "/helperfunctions")
from preprocfunc import OutlierTrans

pd.set_option('display.width', 78)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.2f}'.format

# setup the features and target
healthinfo = pd.read_csv("data/healthinfosample.csv")

healthinfo['diabetic'] = \
  np.where(healthinfo.diabetic.str[0:2]=='No','No','Yes')
healthinfo['heartdisease'] = \
  np.where(healthinfo.heartdisease=='No',0,1).astype('int')
  
healthinfo.heartdisease.value_counts()

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
  StandardScaler())
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

new_cols


# add feature selection and a linear model to the pipeline and look at the parameter estimates

svc = SVC(kernel='rbf', class_weight='balanced')

knnimp = KNNImputer(n_neighbors=45)

kf = StratifiedKFold(n_splits=7, shuffle=True, random_state=0)

pipe1 = make_pipeline(coltrans, KNNImputer(n_neighbors=5), svc)

svc_params = {
 'svc__C': uniform(loc=0, scale=20),
 'svc__gamma': uniform(loc=0, scale=100)
 }

rs = RandomizedSearchCV(pipe1, svc_params, cv=kf, scoring='roc_auc')
rs.fit(X_train, y_train.values.ravel())

rs.best_params_
rs.best_score_


# get predictions and residuals
pred = rs.predict(X_test)

preddf = pd.DataFrame(pred, columns=['prediction'],
  index=X_test.index).join(X_test).join(y_test)

preddf['resid'] = preddf.avgtemp-preddf.prediction

plt.scatter(preddf.prediction, preddf.resid, color="blue")
plt.axhline(0, color='red', linestyle='dashed', linewidth=1)
plt.title("Scatterplot of Predictions and Residuals")
plt.xlabel("Predicted Gax Tax")
plt.ylabel("Residuals")
plt.show()


pd.DataFrame(np.logspace(0, 4, 10), columns=['values']).to_excel('views/test.xlsx')

uniform(loc=0, scale=4).rvs(10)
uniform(loc=0.1, scale=2.0).rvs(100)

# plot the residuals
plt.hist(preddf.resid, color="blue", bins=np.arange(-0.5,1.0,0.25))
plt.axvline(preddf.resid.mean(), color='red', linestyle='dashed', linewidth=1)
plt.title("Histogram of Residuals for Gax Tax Model")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.xlim()
plt.show()

# plot predictions against the residuals
plt.scatter(preddf.prediction, preddf.resid, color="blue")
plt.axhline(0, color='red', linestyle='dashed', linewidth=1)
plt.title("Scatterplot of Predictions and Residuals")
plt.xlabel("Predicted Gax Tax")
plt.ylabel("Residuals")
plt.show()


# do kfold cross validation
X_train, X_test, y_train, y_test =  \
  train_test_split(features,\
  target, test_size=0.1, random_state=22)

kf = KFold(n_splits=3, shuffle=True, random_state=0)

cross_validate(ttr, X=X_train, y=y_train,
  cv=kf, scoring=('r2', 'neg_mean_absolute_error'), n_jobs=-1)

