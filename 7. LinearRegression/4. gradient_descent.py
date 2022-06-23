# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import KNNImputer

from sklearn.model_selection import GridSearchCV

import os
import sys
sys.path.append(os.getcwd() + "/helperfunctions")
from preprocfunc import OutlierTrans

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.2f}'.format

fftaxrate14 = pd.read_csv("data/fossilfueltaxrate14.csv")
fftaxrate14.set_index('countrycode', inplace=True)
fftaxrate14.info()

# setup the features and target
num_cols = ['fuel_income_dependence','national_income_per_cap',
  'VAT_Rate',  'gov_debt_per_gdp','polity','goveffect',
  'democracy_index']
dummy_cols = ['democracy_polity','autocracy_polity','democracy',
  'nat_oil_comp','nat_oil_comp_state']
spec_cols = ['motorization_rate']

# generate some summary statistics
fftaxrate14[['gas_tax_imp'] + num_cols + spec_cols].\
  agg(['count','min','median','max']).T
fftaxrate14[dummy_cols].apply(pd.value_counts, normalize=True).T

target = fftaxrate14[['gas_tax_imp']]
features = fftaxrate14[num_cols + dummy_cols + spec_cols]

X_train, X_test, y_train, y_test =  \
  train_test_split(features,\
  target, test_size=0.2, random_state=0)
      
# setup pipelines for column transformation
standtrans = make_pipeline(OutlierTrans(2), SimpleImputer(strategy="median"),
  StandardScaler())
cattrans = make_pipeline(SimpleImputer(strategy="most_frequent"))
spectrans = make_pipeline(OutlierTrans(2), StandardScaler())
coltrans = ColumnTransformer(
  transformers=[
    ("stand", standtrans, num_cols),
    ("cat", cattrans, dummy_cols),
    ("spec", spectrans, spec_cols)
  ]
)

# add feature selection and a linear model to the pipeline and look at the parameter estimates

sgdr_params = {
 'regressor__sgdregressor__alpha': 10.0 ** -np.arange(1, 7),
 'regressor__sgdregressor__loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
 'regressor__sgdregressor__penalty': ['l2', 'l1', 'elasticnet'],
 'regressor__sgdregressor__learning_rate': ['constant', 'optimal', 'invscaling']
}


sgdr = SGDRegressor()

pipe1 = make_pipeline(coltrans, KNNImputer(n_neighbors=5), sgdr)

ttr=TransformedTargetRegressor(regressor=pipe1,transformer=StandardScaler())

gs = GridSearchCV(ttr,param_grid=sgdr_params, cv=5, scoring="r2")
gs.fit(X_train, y_train)

gs.best_params_
gs.best_score_


