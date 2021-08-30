# import pandas and numpy, and load the nls97 data
import pandas as pd
pd.set_option('display.width', 70)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 15)
pd.options.display.float_format = '{:,.0f}'.format
nls97 = pd.read_csv("data/nls97.csv")
nls97.set_index("personid", inplace=True)

# select a column using the pandas index operator
analysisdemo = nls97['gender']
type(analysisdemo)
analysisdemo = nls97[['gender']]
type(analysisdemo)
analysisdemo.dtypes
analysisdemo = nls97.loc[:,'gender']
type(analysisdemo)
analysisdemo = nls97.loc[:,['gender']]
type(analysisdemo)
analysisdemo.dtypes
analysisdemo = nls97.iloc[:,[0]]
type(analysisdemo)
analysisdemo.dtypes

# select multiple columns from a pandas data frame
analysisdemo = nls97[['gender','highestgradecompleted',
  'maritalstatus']]
analysisdemo.dtypes

analysisdemo = nls97.loc[:,['gender','highestgradecompleted',
 'maritalstatus']]
analysisdemo.dtypes

analysisdemo = nls97.iloc[:,[0,3,4]]
analysisdemo.dtypes


# use lists to select multiple columns
keyvars = ['gender','maritalstatus',
 'highestgradecompleted','wageincome',
 'gpaoverall','weeksworked17','colenroct17']
analysiskeys = nls97[keyvars]
analysiskeys.dtypes

# select multiple columns using the filter operator
analysiswork = nls97.filter(like="weeksworked")
analysiswork.dtypes

# select multiple columns based on data types
analysisobj = nls97.select_dtypes(include=["object"])
analysisobj.dtypes

analysisnotobj = nls97.select_dtypes(exclude=["object"])
analysisnotobj.dtypes

# organize columns
demo = ['gender','birthmonth','birthyear']
highschoolrecord = ['satverbal','satmath','gpaoverall',
 'gpaenglish','gpamath','gpascience']
demoadult = ['highestgradecompleted','maritalstatus',
  'childathome','childnotathome','wageincome',
  'weeklyhrscomputer','weeklyhrstv','nightlyhrssleep',
  'highestdegree']

nls97 = nls97[demoadult + demo + highschoolrecord]
nls97.dtypes

nls97 = nls97.loc[:, demoadult + demo + highschoolrecord]


