# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skmet

import matplotlib.pyplot as plt

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
coltrans = ColumnTransformer(
  transformers=[
    ("bin", bintrans, binary_cols),
    ("cat", cattrans, cat_cols),
  ],
    remainder = 'passthrough'
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

# construct a pipeline with preprocessing, feature selection, and logistic model
catcolscnt = new_binary_cols.shape[0] + new_cat_cols.shape[0]
smotenc = SMOTENC(categorical_features=np.arange(0,catcolscnt), random_state=0)


# do some hyperparameter tuning
rfc = RandomForestClassifier(random_state=0)

pipe1 = make_pipeline(coltrans, smotenc, rfc)

rfc_params = {
 'randomforestclassifier__min_samples_leaf': np.arange(100, 1200, 100),
 'randomforestclassifier__max_depth': np.arange(2, 11)
}


rs = RandomizedSearchCV(pipe1, rfc_params, cv=5, scoring="roc_auc")
rs.fit(X_train, y_train.values.ravel())

rs.best_params_
rs.best_score_

pred = rs.predict(X_test)

accuracy, sensitivity, specificity, precision = \
  skmet.accuracy_score(y_test.values.ravel(), pred),\
  skmet.recall_score(y_test.values.ravel(), pred),\
  skmet.recall_score(y_test.values.ravel(), pred,  pos_label=0),\
  skmet.precision_score(y_test.values.ravel(), pred)
accuracy, sensitivity, specificity, precision


cm = skmet.confusion_matrix(y_test, pred)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
cmplot.plot()
cmplot.ax_.set(title='Heart Disease Prediction Confusion Matrix', 
  xlabel='Predicted Value', ylabel='Actual Value')
