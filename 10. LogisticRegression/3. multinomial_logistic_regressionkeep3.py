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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import TransformedTargetRegressor
from scipy.stats import uniform

#from sklearn.metrics import accuracy_score
import sklearn.metrics as skmet

import os
import sys
sys.path.append(os.getcwd() + "/helperfunctions")
from preprocfunc import OutlierTrans

pd.set_option('display.width', 78)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.0f}'.format

# load the health information data
nls97degreelevel = pd.read_csv("data/nls97degreelevel.csv")
nls97degreelevel.info()
nls97degreelevel.head()
nls97degreelevel.failtype.value_counts(dropna=False).sort_index()
nls97degreelevel.machinetype.value_counts(dropna=False).sort_index()

nls97degreelevel.groupby(['failtypecode','failtype']).size().\
  reset_index()
  
# take a look at some of the data

# identify numeric and categorical data
num_cols = ['satverbal','satmath','gpascience','gpaenglish','gpamath',
  'gpaoverall','motherhighgrade','fatherhighgrade','parentincome']
binary_cols = ['gender']

nls97degreelevel[num_cols].agg(['min','median','max']).T

# create training and testing DataFrames
X_train, X_test, y_train, y_test =  \
  train_test_split(nls97degreelevel[num_cols + binary_cols],\
  nls97degreelevel[['degreelevel']], test_size=0.2, random_state=0)


# setup column transformations
ohe = OneHotEncoder(drop='first', sparse=False)

standtrans = make_pipeline(OutlierTrans(3),SimpleImputer(strategy="median"),
  StandardScaler())
bintrans = make_pipeline(ohe)
cattrans = make_pipeline(ohe)
coltrans = ColumnTransformer(
  transformers=[
    ("stand", standtrans, num_cols),
    ("bin", bintrans, binary_cols)
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
