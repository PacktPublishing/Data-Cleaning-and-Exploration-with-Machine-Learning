# import pandas, numpy, matplotlib, statsmodels, and load the covid totals data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
pd.set_option('display.width', 85)
pd.options.display.float_format = '{:,.3f}'.format
covidtotals = pd.read_csv("data/covidtotals.csv")
covidtotals.set_index("iso_code", inplace=True)
covidtotals.info()

# create an analysis file
xvars = ['population_density','aged_65_older','gdp_per_capita','diabetes_prevalence']

covidanalysis = covidtotals.loc[:,['total_cases_mill'] + xvars].dropna()
covidanalysis.describe()

# fit a linear regression model
def getlm(df, ycolname, xcolnames):
  df = df[[ycolname] + xcolnames].dropna()
  y = df[ycolname]
  X = df[xcolnames]
  X = sm.add_constant(X)
  lm = sm.OLS(y, X).fit()
  coefficients = pd.DataFrame(zip(['constant'] + xcolnames, lm.params, lm.pvalues), columns=['features','params','pvalues'])
  return lm, coefficients

lm, coefficients = getlm(covidtotals, 'total_cases_mill', ['population_density','aged_65_older','gdp_per_capita','diabetes_prevalence'])
lm.summary()
coefficients

# identify countries with an outsized influence on the model
influence = lm.get_influence().summary_frame()
influencethreshold = 3*influence.cooks_d.mean()
covidtotals = covidtotals.join(influence[['cooks_d']])
covidtotalsminusoutliers = covidtotals.loc[covidtotals.cooks_d>=influencethreshold,
  ['location','total_cases_mill','cooks_d'] + xvars].\
  sort_values(['cooks_d'], ascending=False)
covidtotalsminusoutliers.head()
covidtotalsminusoutliers.shape

lm = getlm(covidtotalsminusoutliers, 'total_cases_mill', ['population_density','aged_65_older','gdp_per_capita','diabetes_prevalence'])
lm.summary()

# do an influence plot
fig, ax = plt.subplots()
sm.graphics.influence_plot(lm, ax = ax, criterion="cooks")
plt.show()

# show a model without the outliers
covidanalysisminusoutliers = covidanalysis.loc[influence.cooks_d<0.5]

lm = getlm(covidanalysisminusoutliers)
lm.summary()

nls97 = pd.read_csv("data/nls97.csv")
nls97.set_index("personid", inplace=True)

nls97.info()
nls97['earnedba'] = np.where(nls97.highestdegree.isnull(), np.nan, np.where(nls97.highestdegree.str[0:1].isin(['4','5','6','7']),1,0))
nls97['female'] = np.where(nls97.gender.str.strip()=="Female",1,0)
nls97['evermarried'] = np.where(nls97.maritalstatus.isnull(),
  np.nan,np.where(nls97.maritalstatus.str.strip()=="Never-married",0,1))
lm = getlm(nls97, 'weeksworked17', ['earnedba','childathome','evermarried','female'])
lm.summary()
influence = lm.get_influence().summary_frame()
influencethreshold = 3*influence.cooks_d.mean()

nls97 = nls97.join(influence[['cooks_d']])
nls97minusoutliers = nls97.loc[nls97.cooks_d>=influencethreshold, ['weeksworked17','earnedba','childathome','evermarried','female','cooks_d']].sort_values(['cooks_d'], ascending=False)
nls97minusoutliers.head(10)
lm = getlm(nls97minusoutliers, 'weeksworked17', ['earnedba','childathome','evermarried','female'])
lm.summary()

