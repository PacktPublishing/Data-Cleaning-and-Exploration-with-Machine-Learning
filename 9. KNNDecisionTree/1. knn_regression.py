# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest,f_regression
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.getcwd() + "/helperfunctions")
from preprocfunc import OutlierTrans

pd.set_option('display.width', 78)
pd.set_option('display.max_columns', 12)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.2f}'.format

# load the land temperatures data
un_income_gap = pd.read_csv("data/un_income_gap.csv")
un_income_gap.set_index('country', inplace=True)
un_income_gap['incomegap'] = \
  un_income_gap.maleincomepercapita - un_income_gap.femaleincomepercapita
un_income_gap['educgap'] = \
  un_income_gap.maleyearseducation - un_income_gap.femaleyearseducation
un_income_gap['laborforcepartgap'] = \
  un_income_gap.malelaborforceparticipation - un_income_gap.femalelaborforceparticipation
un_income_gap['humandevgap'] = \
  un_income_gap.malehumandevelopment - un_income_gap.femalehumandevelopment
un_income_gap.dropna(subset=['incomegap'], inplace=True)


num_cols = ['educgap','laborforcepartgap','humandevgap',
  'genderinequality','maternalmortaility','adolescentbirthrate',
  'femaleperparliament','incomepercapita']

un_income_gap[num_cols].head()


un_income_gap[['incomegap'] + num_cols].\
  agg(['count','min','median','max']).T


# create training and testing DataFrames
X_train, X_test, y_train, y_test =  \
  train_test_split(un_income_gap[num_cols],\
  un_income_gap[['incomegap']], test_size=0.2, random_state=0)


# construct a pipeline with preprocessing, feature selection, and knn model
knnreg = KNeighborsRegressor()

pipe1 = make_pipeline(OutlierTrans(3), SimpleImputer(strategy="median"),
  StandardScaler(), SelectKBest(score_func=f_regression, k=5), knnreg)

ttr=TransformedTargetRegressor(regressor=pipe1,
  transformer=StandardScaler())

knnreg_params = {
 'regressor__selectkbest__k': np.arange(1, 8),
 'regressor__kneighborsregressor__n_neighbors': np.arange(3, 21, 2),
 'regressor__kneighborsregressor__metric': ['euclidean','manhattan','minkowski']
}

# do a randmoized parameter search
rs = RandomizedSearchCV(ttr, knnreg_params, cv=10, scoring='r2')
rs.fit(X_train, y_train)

rs.best_params_
rs.best_score_


# get predictions and residuals
pred = rs.predict(X_test)

preddf = pd.DataFrame(pred, columns=['prediction'],
  index=X_test.index).join(X_test).join(y_test)

preddf['resid'] = preddf.incomegap-preddf.prediction


plt.hist(preddf.resid, color="blue")
plt.axvline(preddf.resid.mean(), color='red', linestyle='dashed', linewidth=1)
plt.title("Histogram of Residuals for Income Gap Model")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.xlim()
plt.show()




plt.scatter(preddf.prediction, preddf.resid, color="blue")
plt.axhline(0, color='red', linestyle='dashed', linewidth=1)
plt.title("Scatterplot of Predictions and Residuals")
plt.xlabel("Predicted Income Gap")
plt.ylabel("Residuals")
plt.show()

preddf.head(10)


