# import pandas, numpy, and matplotlib
import pandas as pd
from feature_engine.encoding import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from feature_engine.discretisation import EqualFrequencyDiscretiser as efd
pd.set_option('display.width', 75)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)
pd.options.display.float_format = '{:,.3f}'.format
nls97compba = pd.read_csv("data/nls97compba.csv")

feature_cols = ['gender','satverbal','satmath','gpascience',
  'gpaenglish','gpamath','gpaoverall','motherhighgrade',
  'fatherhighgrade','parentincome']

# separate NLS data into train and test datasets
X_train, X_test, y_train, y_test =  \
  train_test_split(nls97compba[feature_cols],\
  nls97compba[['completedba']], test_size=0.3, random_state=0)

# encode the gender feature and scale the other features
ohe = OneHotEncoder(drop_last=True, variables=['gender'])
X_train_enc = ohe.fit_transform(X_train)
scaler = StandardScaler()
standcols = X_train_enc.iloc[:,:-1].columns
X_train_enc = \
  pd.DataFrame(scaler.fit_transform(X_train_enc[standcols]),
  columns=standcols, index=X_train_enc.index).\
  join(X_train_enc[['gender_Female']])

# select 5 best features for predicting college completion using mutual information
ksel = SelectKBest(score_func=mutual_info_classif, k=5)
ksel.fit(X_train_enc, y_train.values.ravel())
selcols = X_train_enc.columns[ksel.get_support()]
selcols
pd.DataFrame({'score': ksel.scores_,
  'feature': X_train_enc.columns},
   columns=['feature','score']).\
   sort_values(['score'], ascending=False)
X_train_analysis = X_train_enc[selcols]
X_train_analysis.dtypes

# set up bins based on equal frequency
bintransformer = efd(q=5, variables=feature_cols[1:])
bintransformer.fit(X_train_enc)
X_train_bin = bintransformer.transform(X_train_enc)

# select 5 best features for predicting college completion using chi-square
ksel = SelectKBest(score_func=chi2, k=5)
ksel.fit(X_train_bin, y_train)
selcols = X_train_enc.columns[ksel.get_support()]
selcols
pd.DataFrame({'score': ksel.scores_,
  'feature': X_train_enc.columns},
   columns=['feature','score']).\
   sort_values(['score'], ascending=False)



