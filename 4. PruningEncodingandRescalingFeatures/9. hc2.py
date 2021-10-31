# import pandas, numpy, and matplotlib
import pandas as pd
from feature_engine.encoding import OneHotEncoder
from category_encoders.hashing import HashingEncoder
from sklearn.model_selection import train_test_split
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.0f}'.format

covidtotals = pd.read_csv("data/covidtotals.csv")
feature_cols = ['location','population',
    'aged_65_older','diabetes_prevalence','region']
covidtotals = covidtotals[['total_cases'] + feature_cols].dropna()

# Separate into train and test sets
X_train, X_test, y_train, y_test =  \
  train_test_split(covidtotals[feature_cols],\
  covidtotals[['total_cases']], test_size=0.3, random_state=0)

# use the one hot encoder for region
X_train.region.value_counts()
ohe = OneHotEncoder(top_categories=6, variables=['region'])
covidtotals_ohe = ohe.fit_transform(covidtotals)
covidtotals_ohe.filter(regex='location|region',
  axis="columns").sample(5, random_state=99).T

# use the hashing encoder for region
he = HashingEncoder(cols=['region'], n_components=16)
covidtotals['region2'] = covidtotals.region
covidtotals_enc = he.fit_transform(covidtotals)

covidtotals_enc.filter(regex='col|reg', axis="columns")
covidtotals_enc.groupby(['col_0','col_1','col_2','col_3','col_4','col_5','col_6','col_7','col_8','col_9','col_10','col_11','col_12','col_13','col_14','col_15','region2']).size().reset_index()

