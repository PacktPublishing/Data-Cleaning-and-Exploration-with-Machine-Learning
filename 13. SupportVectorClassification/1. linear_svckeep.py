# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from scipy.stats import uniform
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer



from sklearn.model_selection import cross_validate, \
  RandomizedSearchCV, StratifiedKFold
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

# fit an SVR model
svc = LinearSVC(max_iter=1000000, class_weight='balanced', random_state=0)

pipe1 = make_pipeline(coltrans, KNNImputer(n_neighbors=5), svc)

pipe1.fit(X_train, y_train.values.ravel())

coefs = pipe1['linearsvc'].coef_
coefwithlabs = np.column_stack((coefs.ravel(), new_cols))
np.sort(coefwithlabs, axis=0)[::-1]
pipe1['linearsvc'].intercept_
pipe1['linearsvc'].support_


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
 'linearsvc__C': uniform(loc=0, scale=100)
}

rs = RandomizedSearchCV(pipe1, svc_params, cv=kf, scoring='roc_auc')
rs.fit(X_train, y_train.values.ravel())

rs.best_params_

rs.best_score_


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

