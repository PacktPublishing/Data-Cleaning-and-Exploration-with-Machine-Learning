# import pandas and scikit learn's KNNImputer module
import pandas as pd
import numpy as np
import scipy.sparse as sp
pip install datawig
import datawig
pd.options.display.float_format = '{:,.1f}'.format
nls97 = pd.read_csv("data/nls97b.csv")
nls97.set_index("personid", inplace=True)

# load the NLS wage data
nls97['hdegnum'] = nls97.highestdegree.str[0:1].astype('float')
nls97.parentincome.replace(list(range(-5,0)), np.nan, inplace=True)
nls97['degltcol'] = np.where(nls97.hdegnum<=2,1,0)

nls97['degcol'] = np.where(nls97.hdegnum.between(3,4),1,0)
nls97['degadv'] = np.where(nls97.hdegnum>4,1,0)

wagedatalist = ['wageincome','weeksworked16','parentincome',
  'degltcol','degcol','degadv']
wagedatalistimp = ['wageincomeimp','weeksworked16imp','parentincomeimp',
  'degltcol','degcol','degadv']
wagedata = nls97[wagedatalist]

wagedatalist[1:]

# initialize a KNN imputation model and fill values
wd_train, wd_test = dwig.utils.random_split(wagedata)

#Initialize a SimpleImputer model
imputer = dwig.SimpleImputer(
    input_columns=wagedatalist[1:], 
    output_column= 'wageincome',
    output_path = 'imputer_model'
    )

#Fit an imputer model on the train data
imputer.fit(train_df=df_train, num_epochs=50)

#Impute missing values and return original dataframe with predictions
imputed = imputer.predict(df_test)

# view imputed values
wagedata = wagedata.join(wagedataimp[['wageincomeimp','weeksworked16imp']])
wagedata[['wageincome','weeksworked16','parentincome',
  'degcol','degadv','wageincomeimp']].head(10)

wagedata[['wageincome','wageincomeimp','weeksworked16','weeksworked16imp']].\
  agg(['count','mean','std'])

