# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from imblearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTENC
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2

import sklearn.metrics as skmet
from sklearn.model_selection import cross_validate
import os
import sys
sys.path.append(os.getcwd() + "/helperfunctions")
from preprocfunc import OutlierTrans

pd.set_option('display.width', 78)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.3f}'.format

# load the health information data
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
standtrans = make_pipeline(OutlierTrans(3),
  SimpleImputer(strategy="median"),
  MinMaxScaler())
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

# construct a pipeline with preprocessing, feature selection, and logistic model
catcolscnt = new_binary_cols.shape[0] + new_cat_cols.shape[0]
smotenc = SMOTENC(categorical_features=np.arange(0,catcolscnt), random_state=0)

knn_example = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

pipe0 = make_pipeline(coltrans, smotenc, knn_example)

#pipe0.fit(X_train, y_train.values.ravel())

kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)

scores = cross_validate(pipe0, X_train, y_train.values.ravel(), \
  scoring=['accuracy','precision','recall','f1'], cv=kf, n_jobs=-1)


print("accuracy: %.2f, precision: %.2f, sensitivity: %.2f, f1: %.2f"  %
  (np.mean(scores['test_accuracy']),\
  np.mean(scores['test_precision']),\
  np.mean(scores['test_recall']),\
  np.mean(scores['test_f1'])))


# do some hyperparameter tuning
knn = KNeighborsClassifier(n_jobs=-1)

pipe1 = make_pipeline(coltrans, smotenc, SelectKBest(score_func=chi2), knn)

knn_params = {
 'selectkbest__k': np.arange(1, len(new_cols)),
 'kneighborsclassifier__n_neighbors': np.arange(5, 303, 2),
 'kneighborsclassifier__metric': ['euclidean','manhattan','minkowski']
}

rs = RandomizedSearchCV(pipe1, knn_params, cv=5, scoring="roc_auc")
rs.fit(X_train, y_train.values.ravel())

selected = rs.best_estimator_['selectkbest'].get_support()
selected.sum()
new_cols[selected]
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
