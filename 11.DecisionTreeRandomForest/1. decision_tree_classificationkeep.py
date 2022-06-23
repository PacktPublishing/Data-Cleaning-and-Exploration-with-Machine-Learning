# import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
# from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFECV
#import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import export_graphviz


#from sklearn.metrics import accuracy_score
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import seaborn as sb

pd.set_option('display.width', 78)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.3f}'.format

# load the health information data
healthinfo = pd.read_csv("data/healthinfo.csv")

healthinfo['diabetic'] = \
  np.where(healthinfo.diabetic.str[0:2]=='No','No','Yes')
healthinfo['heartdisease'] = \
  np.where(healthinfo.heartdisease=='No',0,1).astype('int')
  
healthinfo.genhealth.value_counts()

# take a look at some of the data

# identify numeric and categorical data
num_cols = ['bmi','physicalhealthbaddays','mentalhealthbaddays',
  'sleeptimenightly']
binary_cols = ['smoking','alcoholdrinkingheavy','stroke',
  'walkingdifficult','diabetic','physicalactivity','asthma',
  'kidneydisease','skincancer']
cat_cols = ['gender','agecategory','ethnicity','genhealth']

# create training and testing DataFrames
X_train, X_test, y_train, y_test =  \
  train_test_split(healthinfo[num_cols + binary_cols + cat_cols],\
  healthinfo[['heartdisease']], test_size=0.2, random_state=0)


# setup column transformations
ohe = OneHotEncoder(drop='first', sparse=False)
rus = RandomUnderSampler(random_state=0)

catcols = np.concatenate((new_binary_cols, new_cat_cols))
catcols.shape

np.arange(0,30)

coltrans.fit(X_train.sample(1000))
catcolcnt = coltrans.named_transformers_['cat'].named_steps['onehotencoder'].get_feature_names_out(cat_cols).shape[0] + \
  coltrans.named_transformers_['bin'].named_steps['onehotencoder'].get_feature_names_out(binary_cols).shape[0]
catcolcnt
coltrans.named_transformers_['bin'].named_steps['onehotencoder'].get_feature_names_out(binary_cols)
len(binary_cols)


smotenc = SMOTENC(categorical_features=np.arange(0,catcolcnt), random_state=0)
smotenc.fit_resample(xtemp2, y_train.values.ravel())
xtemp2 = coltrans.fit_transform(X_train)
xt2 = pd.DataFrame(xtemp2, columns=new_cols)
xt2.head().T
xt2[catcols].agg(['min','median','max']).T
catcols


bintrans = make_pipeline(ohe)
cattrans = make_pipeline(ohe)
coltrans = ColumnTransformer(
  transformers=[
    ("bin", bintrans, binary_cols),
    ("cat", cattrans, cat_cols),
  ],
    remainder = 'passthrough'
)

pipe2 = make_pipeline(coltrans, smotenc)
healthinfo.head().T

xtemp, ytemp = pipe2.fit_resample(X_train, y_train.values.ravel())

xt = pd.DataFrame(xtemp, columns=new_cols)

xt.groupby(['walkingdifficult_Yes','genhealth_Very good']).size().reset_index()

xt.walkingdifficult_Yes.value_counts()
xt["genhealth_Very good"].value_counts()
xt2.gender_Male.value_counts().shape

# construct a pipeline with preprocessing, feature selection, and logistic model
# construct a pipeline with preprocessing and knn model
dtc_example = DecisionTreeClassifier(min_samples_leaf=5,
  max_depth=3)

pipe0 = make_pipeline(coltrans, rus, dtc_example)
pipe0 = make_pipeline(coltrans, smotenc, dtc_example)

pipe0.fit(X_train, y_train.values.ravel())

new_binary_cols = \
  pipe0.named_steps['columntransformer'].\
  named_transformers_['bin'].\
  named_steps['onehotencoder'].\
  get_feature_names_out(binary_cols)
new_cat_cols = \
  pipe0.named_steps['columntransformer'].\
  named_transformers_['cat'].\
  named_steps['onehotencoder'].\
  get_feature_names_out(cat_cols)

new_cols = np.concatenate((new_binary_cols, new_cat_cols, np.array(num_cols)))

new_cols

#dot_data = export_graphviz(pipe0.named_steps['decisiontreeclassifier'],
  #out_file ='tree.dot', feature_names=new_cols)

plot_tree(pipe0.named_steps['decisiontreeclassifier'], 
  feature_names=new_cols, class_names=['No Disease','Disease'], fontsize=10)

healthinfo.head()

lrsel = LogisticRegression(random_state=0, max_iter=1000)

kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)

rfecv = RFECV(estimator=lrsel, cv=kf)

pipe1 = make_pipeline(coltrans, rus, DecisionTreeClassifier())
pipe1

dtc_params = {
 'decisiontreeclassifier__min_samples_leaf': np.arange(1,8),
 'decisiontreeclassifier__max_depth': np.arange(1,8)
}

rs = RandomizedSearchCV(pipe1, reg_params, cv=5, scoring='roc_auc')
rs.fit(X_train, y_train.values.ravel())

rs.best_params_
rs.best_score_



lr = LogisticRegression(random_state=0, class_weight='balanced',
  max_iter=1000)

pipe1 = make_pipeline(coltrans, rfecv, lr)

pipe1.fit(X_train, y_train.values.ravel())

# get the columnn names from the onehotencoder


# look at the rankings from the recursive featue elimination

rankinglabs = np.column_stack((pipe1.named_steps['rfecv'].ranking_, new_cols))
np.sort(rankinglabs, axis=0)

# get the coefficients from the logistic regression
oddsratios = np.exp(pipe1.named_steps['logisticregression'].coef_)
oddsratios.shape
selcols = new_cols[pipe1.named_steps['rfecv'].get_support()]
oddswithlabs = np.column_stack((oddsratios.ravel(), selcols))
np.sort(oddswithlabs, axis=0)[::-1]


# get predictions and residuals
pred = pipe1.predict(X_test)

cm = skmet.confusion_matrix(y_test, pred)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
cmplot.plot()
cmplot.ax_.set(title='Heart Disease Prediction Confusion Matrix', 
  xlabel='Predicted Value', ylabel='Actual Value')


tn, fp, fn, tp = skmet.confusion_matrix(y_test.values.ravel(), pred).ravel()
tn, fp, fn, tp
accuracy = (tp + tn) / pred.shape[0]
accuracy
sensitivity = tp / (tp + fn)
sensitivity
specificity = tn / (tn+fp)
specificity
precision = tp / (tp + fp)
precision

# another way to calculate the metrics
accuracy, sensitivity, specificity, precision = \
  skmet.accuracy_score(y_test.values.ravel(), pred),\
  skmet.recall_score(y_test.values.ravel(), pred),\
  skmet.recall_score(y_test.values.ravel(), pred,  pos_label=0),\
  skmet.precision_score(y_test.values.ravel(), pred)
accuracy, sensitivity, specificity, precision



falsepositiverate = fp / (tn + fp)
falsepositiverate

# do a density plot
pred_probs = pipe1.predict_proba(X_test)[:, 1]

probdf = \
  pd.DataFrame(zip(pred_probs, pred, y_test.values.ravel()),
  columns=(['prob','pred','actual']))

probdf.groupby(['pred'])['prob'].\
  agg(['min','max','count'])


sb.kdeplot(probdf.loc[probdf.actual==1].prob, shade=True, color='red',
  label="Heart Disease")
sb.kdeplot(probdf.loc[probdf.actual==0].prob, shade=True, color='green',
  label="No Heart Disease")
plt.axvline(0.25, color='black', linestyle='dashed', linewidth=1)
plt.axvline(0.5, color='black', linestyle='dashed', linewidth=1)
plt.title("Predicted Probability Distribution")
plt.legend(loc="upper left")




# plot precision and sensitivity curve
sens, prec, ths = skmet.precision_recall_curve(y_test, pred_probs)

# plot precision and sensitivity lines
sens, prec, ths = skmet.precision_recall_curve(y_test, pred_probs)
sens = sens[1:-20]
prec = prec[1:-20]
ths  = ths[:-20]

fig, ax = plt.subplots()
ax.plot(ths, prec, label='Precision')
ax.plot(ths, sens, label='Sensitivity')
ax.set_title('Precision and Sensitivity by Threshold')
ax.set_xlabel('Threshold')
ax.set_ylabel('Precision and Sensitivity')
#ax.set_xlim(0.3,0.9)
ax.legend()

# plot ROC curve
fpr, tpr, ths = skmet.roc_curve(y_test, pred_probs)
ths = ths[1:]
fpr = fpr[1:]
tpr = tpr[1:]

fig, ax = plt.subplots()
ax.plot(fpr, tpr, linewidth=4, color="black")
ax.set_title('ROC curve')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('Sensitivity')

fig, ax = plt.subplots()
ax.plot(ths, fpr, label="False Positive Rate")
ax.plot(ths, tpr, label="Sensitivity")
ax.set_title('False Positive Rate and Sensitivity by Threshold')
ax.set_xlabel('Threshold')
ax.set_ylabel('False Positive Rate and Sensitivity')
ax.legend()


jthresh = ths[np.argmax(tpr - fpr)]
jthresh
tpr.shape

fscore = (2 * precision * sensitivity) / (precision + sensitivity)
fthresh = ths[np.argmax(fscore)]
fthresh


skmet.precision_recall_threshold(sens, prec, ths, 0.5)

pred2 = np.where(pred_probs>=jthresh,1,0)
cm = skmet.confusion_matrix(y_test, pred2)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
cmplot.plot()
cmplot.ax_.set(title='Heart Disease Prediction Confusion Matrix', 
  xlabel='Predicted Value', ylabel='Actual Value')


skmet.recall_score(y_test.values.ravel(), pred)
skmet.recall_score(y_test.values.ravel(), pred2)
