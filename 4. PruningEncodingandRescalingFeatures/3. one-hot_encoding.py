# import pandas, numpy, and matplotlib
import pandas as pd
from feature_engine.encoding import OneHotEncoder
from feature_engine.encoding import OrdinalEncoder
from sklearn.model_selection import train_test_split
pd.set_option('display.width', 80)
pd.set_option('display.max_columns', 8)
pd.set_option('display.max_rows', 100)
pd.options.display.float_format = '{:,.0f}'.format
nls97 = pd.read_csv("data/nls97b.csv")
nls97.set_index("personid", inplace=True)
coviddaily = pd.read_csv("data/coviddaily.csv", parse_dates=["casedate"])

feature_cols = ['gender','maritalstatus','colenroct99']
nls97demo = nls97[['wageincome'] + feature_cols].dropna()

# separate NLS data into train and test datasets
X_demo_train, X_demo_test, y_demo_train, y_demo_test =  \
  train_test_split(nls97demo[feature_cols],\
  nls97demo[['wageincome']], test_size=0.3, random_state=0)

# use get dummies to create dummies features
pd.get_dummies(X_demo_train, columns=['gender','maritalstatus']).head(2).T
pd.get_dummies(X_demo_train, columns=['gender','maritalstatus'],
  drop_first=True).head(2).T

# use the one hot encoder to create encoded features for gender and marital status
ohe = OneHotEncoder(drop_last=True, variables=['gender','maritalstatus'])
ohe.fit(X_demo_train)
X_demo_train_ohe = ohe.transform(X_demo_train)
X_demo_test_ohe = ohe.transform(X_demo_test)
X_demo_train_ohe.filter(regex='gen|mar', axis="columns").head(2).T

# load the covid data
feature_cols = ['location','population','colenroct99',
    'aged_65_older','diabetes_prevalence','region']

coviddaily = coviddaily[['new_cases'] + feature_cols].dropna().\
  sample(1000, random_state=0)

# separate NLS data into train and test datasets
X_covid_train, X_covid_test, y_covid_train, y_covid_test =  \
  train_test_split(coviddaily[feature_cols], \
  coviddaily[['new_cases']], test_size=0.3, random_state=0)

# use the ordinal encoder for college enrollment
oe = OrdinalEncoder(encoding_method='arbitrary', 
  variables=['colenroct99'])
oe.fit(X_demo_train)
X_demo_train_enc = oe.transform(X_demo_train)
X_demo_test_enc = oe.transform(X_demo_test)
X_demo_train.colenroct99.value_counts().sort_index()
X_demo_train_enc.colenroct99.value_counts().sort_index()

