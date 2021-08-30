# import pandas, numpy, and matplotlib
import pandas as pd
pd.set_option('display.width', 80)
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 100)
pd.options.display.float_format = '{:,.0f}'.format
nls97 = pd.read_csv("data/nls97b.csv")
nls97.set_index("personid", inplace=True)
covidtotals = pd.read_csv("data/covidtotals.csv")
covidtotals.set_index("iso_code", inplace=True)

nlsparents = nls97.iloc[:,-4:]

nlsparents.agg('min')
# %timeit nlsparents.apply(lambda x: x.between(-10,0).value_counts())

nlsparents.loc[nlsparents.motherhighgrade.between(-10,-1), 'motherhighgrade'].value_counts()
nlsparents.motherhighgrade.between(-10,-1).value_counts()
nlsparents.loc[nlsparents.apply(lambda x: x.between(-10,-1)).any(axis=1)]
nlsparents.apply(lambda x: x.between(-10,-1).sum())


# set up the cumulative and demographic columns
totvars = ['location','total_cases_mill','total_deaths_mill']
demovars = ['population_density','aged_65_older',
   'gdp_per_capita','life_expectancy','diabetes_prevalence']

# check the demographic columns for missing
covidtotals[demovars].isnull().sum(axis=0)
demovarsmisscnt = covidtotals[demovars].isnull().sum(axis=1)
demovarsmisscnt.value_counts()
covidtotals.loc[demovarsmisscnt>=3, ['location'] + demovars].head(5).T

# check the cumulative columns for missing
covidtotals[totvars].isnull().sum(axis=0)
totvarsmisscnt = covidtotals[totvars].isnull().sum(axis=1)
totvarsmisscnt.value_counts()
covidtotals.loc[totvarsmisscnt>0].T

# use the fillna method to fix the mixing case data
covidtotals.total_cases_pm. \
  fillna(covidtotals.total_cases/
  (covidtotals.population/1000000),
  inplace=True)
covidtotals.total_deaths_pm. \
  fillna(covidtotals.total_deaths/
  (covidtotals.population/1000000),
  inplace=True)
covidtotals[totvars].isnull().sum(axis=0)



nls97add = pd.read_csv('data/nls97add.csv',
  names=['originalid','motherage','parentincome',
  'fatherhighgrade','motherhighgrade'], skiprows=1)
nls97add.head(5)
nls97add.parentincome.describe()
nls97add.to_pickle("data/nls/nls97add.pkl")

nls97b = pd.merge(nls97, nls97add, left_on=['originalid'], right_on=['originalid'], how="left")
nls97b.set_index("personid", inplace=True)
nls97b.to_csv("data/nls97b.csv")

nls97.dtypes
nls97.index
 