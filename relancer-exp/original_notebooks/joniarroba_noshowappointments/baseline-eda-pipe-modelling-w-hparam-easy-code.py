#!/usr/bin/env python
# coding: utf-8

# # Understanding the No-show Patterns 
# 
# ### Welcome to this kernel. <br>
# 
# First of all, english isn't my first language, so sorry for any mistake.
# ___________________
# ## Objectives and Questions: 
# 
# I will try to understand this data and after some EDA and Data Mining I will build a model to predict the No-show Patients
# 
# Based on the dataset description, I will start with some questions that will guide my exploration.<br>
# Questions like: 
# - The data have missing values? 
# - How many unique values for each column?
# - What are the principal categories for each column?
# - What's the distribution of the target?
# - Whats the distribution of Ages? 
# - We have the same No-show pattern for all age patterns?
# - The Neighbourhood has the same ratio of No-show pattern?
# - What's the range of dates and the distribution of Appointments?
# - And a lot of other questions
# 
# 
# ## NOTE: This kernel is not finished, I am working on it.
# If you think this kernel is usefull, please <b>votesup</b> the kernel 

# ## Importing librarys

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Standard plotly imports
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import iplot, init_notebook_mode
import cufflinks
import cufflinks as cf

# Using plotly + cufflinks in offline mode
init_notebook_mode(connected=True)
cufflinks.go_offline(connected=True)

import gc
import warnings
warnings.filterwarnings("ignore")


# ## Functions
# - If you want to see all functions that I used in this notebook, click in "Show Input" or Fork the Kernel.
# - I will put all functions in the cell below to get a clean notebook.

# In[ ]:


def knowningData(df, limit=5): #seting the function with df, 
    print(f"Dataset Shape: {df.shape}")
    print('Unique values per column: ')
    print(df.nunique())
    print("################")
    print("")    
    for column in df.columns: #initializing the loop
        print("Column Name: ", column )
        entropy = round(stats.entropy(df[column].value_counts(normalize=True), base=2),2)
        print("Entropy ", entropy, " | Total nulls: ", (round(df[column].isnull().sum() / len(df[column]) * 100,2)), " | Total unique values: ", df.nunique()[column],  " | Missing: ", df[column].isna().sum()) 
        print("Top 5 most frequent values: ")
        print(round(df[column].value_counts(normalize=True)[:5],2))
        print("")
        print("####################################")

def object_cols(df, cols):
    for col in cols:
        df[col] = df[col].astype(object)
    return df

def quantiles(df, columns):
    for name in columns:
        print(name + " quantiles")
        print(df[name].quantile([.01,.25,.5,.75,.99]))
        print("")
        
def plotly_plots(df, column, plot_type='bar', title=None, xTitle=None, yTitle=None):
    temp = df[column].value_counts()
    temp.iplot(kind=plot_type, title=title, xTitle=xTitle, yTitle=yTitle)
    
def quantile_plot(x, **kwargs):
    qntls, xr = stats.probplot(x, fit=False)
    plt.scatter(xr, qntls, **kwargs)

# This function is to extract date features
def date_process(df, cols, form=None):
    for col in cols:

        df[col] = pd.to_datetime(df[col]) # seting the column as pandas datetime
        df['_weekdayName_'+str(col)] = df[col].dt.weekday_name #extracting week day
        df['_weekday_'+str(col)] = df[col].dt.weekday #extracting week day        
        df['_day_'+str(col)] = df[col].dt.day # extracting day
        df['_month_'+str(col)] = df[col].dt.month # extracting month
        if col == 'ScheduledDay':
            df['_hour_'+str(col)] = df[col].dt.hour # extracting hour
            df['_minute_'+str(col)] = df[col].dt.minute # extracting minute
        # df[col] = df[col].dt.date.astype('datetime64[ns]')
        
    return df #returning the df after the transformations

def dummies(df, list_cols):
    for col in list_cols:
        df_dummies = pd.get_dummies(df[col], drop_first=True, prefix=(str(col))) 
        df = pd.concat([df, df_dummies], axis=1)
        df.drop(col, axis=1, inplace=True)
        
    return df


# ## Importing datasets

# In[ ]:


df_train = pd.read_csv("../../../input/joniarroba_noshowappointments/KaggleV2-May-2016.csv")


# ## Understanding the date 
# - If you want see the code, look on cells above or fork this Kernel 

# In[ ]:


knowningData(df_train.drop(['PatientId', 'AppointmentID'], axis=1))


# We can see that our data has 110527 rows and 14 columns. <br>
# The data has some category features and we will explore it further later. <br>
# 
# This first view give us a good understanding of the dataset and we can set the name columns to new features, to better filtering 

# ## Semantic lists to better work with each type of cats

# In[ ]:


binary_features = ['Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'SMS_received', 'Gender'] 

target = ['No-show']

categorical = ['Neighbourhood', 'Handcap']

numerical = ['Age']

dates = ['AppointmentDay', 'ScheduledDay']

Ids = ['PatientId', 'AppointmentID']


# # First look at our data

# In[ ]:


df_train.head()


# In[ ]:


## Transforming dates to datetime
df_train = date_process(df_train, dates)


# ## Unique Patients and Unique Appointments Total

# In[ ]:


print(f"Total of Unique Patients is {df_train.PatientId.nunique()} and Appointments is {df_train.AppointmentID.nunique()}")


# Nice. We can see that in average we have almost 2 visits by each patient, that is exactly what we want understand. We will explore it further later

# ## Knowing our target and their distribution

# In[ ]:


plotly_plots(df_train, 'No-show', title='Show and No-Show Distribution', xTitle='No-Show Feature - Target Feature <br>Yes or Not', yTitle='Count') 


# Nice. We can see that we have a high ratio of No-Show patients. It will be very useful to better understand the dataset.<br>
# I will explore the other variables and try to understand the relation between independents and the dependent features
# 

# ## Age features

# In[ ]:


plt.figure(figsize=(16,12))

plt.subplot(221)
g = sns.distplot(df_train['Age'])
g.set_title("Age Count Distribuition", fontsize=18)
g.set_xlabel("")
g.set_ylabel("Probability", fontsize=12)

plt.subplot(222)
g1 = plt.scatter(range(df_train.shape[0]), np.sort(df_train.Age.values))
g1= plt.title("Age ECDF Distribuition", fontsize=18)
g1 = plt.xlabel("Index")
g1 = plt.ylabel("Age Distribution", fontsize=15)

plt.suptitle('Patient Age Distribution', fontsize=22)

print()


# In[ ]:


print(f"The min Age in our data is {df_train.Age.min()} and the max Age is {df_train.Age.max()}")
print(f"Total of Patients with 0 years: {len(df_train[df_train.Age == 0])}")


# In[ ]:


quantiles(df_train, ['Age'])


# Considering the percentiles I will consider only the Ages between 0 to 100 and set the Age to categories to better analize it

# In[ ]:


df_train = df_train[(df_train.Age >= 0 ) & (df_train.Age <= 100)]
print(f"Shape after filtering Data: {df_train.shape}")


# - Cool, we have droped almost 50 rows. <br>
# Now, let's create the age bins

# ## Seting ages to categories
# - I will use the pd.cut function to binarize our data and create categories to Ages

# In[ ]:


bin_ranges = [-1, 2, 8, 16, 18, 25, 40, 50, 60, 75]
bin_names = ["Baby", "Children", "Teenager", 'Young', 'Young-Adult', 'Adult', 'Adult-II', 'Senior', 'Old']

df_train['age_bin'] = pd.cut(np.array(df_train['Age']), bins=bin_ranges, labels=bin_names) 


# ## Age Bins by No-show (target) feature

# In[ ]:


# now stack and reset
show_prob_age = pd.crosstab(df_train['age_bin'], df_train['No-show'], normalize='index')
stacked = show_prob_age.unstack().reset_index().rename(columns={0:'value'})

plt.figure(figsize=(16,12))
plt.subplot(211)
ax1 = sns.countplot(x="age_bin", data=df_train)
ax1.set_title("Age Bins Count", fontsize=22)
ax1.set_xlabel("Age Categories", fontsize=18)
ax1.set_ylabel("Count", fontsize=18)

plt.subplot(212)
ax2 = sns.barplot(x=stacked.age_bin, y=stacked.value, hue=stacked['No-show'])
ax2.set_title("Age Bins by Show and No-show Patients", fontsize=22)
ax2.set_xlabel("Age Categories", fontsize=18)
ax2.set_ylabel("Count", fontsize=18)
ax2.legend(loc='out')

plt.subplots_adjust(hspace = 0.4)

print()


# We can't see a great difference between the categories, altough the we can see that teenager to young-adult has a slightly higher ratio 25% of no-show, against the 22 to 14% of another categories. <br>
# The lower ratios of no-show Patients are the old people and Babys. That make a lot of sense.<br>
# I will explore it further and try to cross this Age categories by some Binary features to see if we can get some insights about the no-show Patients;

# ## Knowing our binary features

# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20,15))
fig.subplots_adjust(hspace=0.3)
fig.suptitle('BINARY FEATURES by the TARGET feature', fontsize=22)

for ax, catplot in zip(axes.flatten(), df_train[binary_features].columns):
    sns.countplot(x=catplot, data=df_train, hue='No-show', ax=ax)
    ax.set_title(catplot.upper(), fontsize=18)
    ax.set_ylabel('Count', fontsize=16)
    ax.set_xlabel(f'{catplot.upper()} Binary Options', fontsize=15)
    ax.legend(title='No-show', fontsize=12, )


# Interesting patterns... We can see that almost all binary's features has the same ratio in the True and False groups. <br>
# In the SMS_Received we can see that the ratio of the True group is very different of the False Group. <br>
# As 

# ## Exploring SMS_received

# In[ ]:


sms_received = df_train.groupby([df_train['AppointmentDay'].dt.date, "SMS_received", "No-show"])['PatientId'].count().reset_index().rename(columns={'PatientId': "Total"}) 


# In[ ]:



plt.figure(figsize=(18,16))

plt.subplot(3,1,1)
g = sns.barplot(x='AppointmentDay', y= 'Total', hue='No-show', data=sms_received[sms_received['SMS_received'] == 0])
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Count of Patients that No Received SMS by No-show feature", fontsize=22)
g.set_xlabel("Dates", fontsize=18)
g.set_ylabel("Count", fontsize=18)

plt.subplot(3,1,2)
g1 = sns.barplot(x='AppointmentDay', y= 'Total', hue='No-show', data=sms_received[sms_received['SMS_received'] == 1])
g1.set_xticklabels(g.get_xticklabels(),rotation=45)
g1.set_title("Count of Patients that Received SMS by No-show feature", fontsize=22)
g1.set_xlabel("Dates", fontsize=18)
g1.set_ylabel("Count", fontsize=18)

plt.subplot(3,1,3)
g2 = sns.boxplot(x='SMS_received', y= 'Age', hue='No-show', data=df_train)
g2.set_xticklabels(g2.get_xticklabels(),rotation=0)
g2.set_title("Received SMS or Not with Age Distribution by No-show feature", fontsize=22)
g2.set_xlabel("Receive SMS or NOT", fontsize=18)
g2.set_ylabel("Age Distribution", fontsize=18)

plt.subplots_adjust(hspace = 0.6)

print()


# Cool!!! My hypothesis is that the many people that received SMS could be to reschedule or cancel the appointment.  <br>
# Also, we can see that people that no-show in appointments has a slightly different Age mean, that is what we saw in the other chart;
# 
# I will explore the binary features below.

# ## Neighbourhood No-show by Age distribution
# - Now we will see the distributions of Age for each Neighbourhood by No-show feature
# - The Neighbourhood with less than 100 entries I will se to "Others"

# In[ ]:


less_than_100 = ['MORADA DE CAMBURI', 'PONTAL DE CAMBURI', 'ILHA DO BOI', 'ILHA DO FRADE', 'AEROPORTO', 'ILHAS OCEÂNICAS DE TRINDADE', 'PARQUE INDUSTRIAL'] 

df_train.loc[df_train.Neighbourhood.isin(less_than_100), 'Neighbourhood'] = "OTHERS"

g = sns.FacetGrid(df_train, col="Neighbourhood", col_wrap=4, size=3, hue='No-show') 

g.map(quantile_plot, "Age").add_legend();
g.set_titles('{col_name}')
print()


# - Nice! Apparently the Neighbourhood don't have some influence in Appointments No-show. 

# ## Date Columns
# - Cleaning
# - Extracting time features
# - Counting Appointments by date
# - Date Distribution

# In[ ]:


## Creating a feature that is the difference between the schedule and the appointment 
df_train['waiting_days'] = (df_train['AppointmentDay'] - df_train['ScheduledDay']).dt.days


# In[ ]:


print(quantiles(df_train, ['waiting_days']))


# Interesting values. It shows that we have many values in -1... We will explore it further later and maybe filter the data that are outliers or any typo.

# ## Range of Dates and the differences between Scheduled and Appointment dates

# In[ ]:


Schedules = (df_train['ScheduledDay'].dt.date.max() - df_train['ScheduledDay'].dt.date.min()).days
Appointments = (df_train['AppointmentDay'].dt.date.max() - df_train['AppointmentDay'].dt.date.min()).days
diff_days = df_train['waiting_days'].max() - df_train['waiting_days'].min() 

print(f"Total date window of SCHEDULES is {Schedules} days. \n        Min date: {df_train['ScheduledDay'].dt.date.min()} \n        Max date: {df_train['ScheduledDay'].dt.date.max()} \n")
print("#"*50, "\n")
print(f"Total date window of APPOINTMENTS is {Appointments} days. \n        Min date: {df_train['AppointmentDay'].dt.date.min()} \n        Max date: {df_train['AppointmentDay'].dt.date.max()}")
print("#"*50, "\n")
print(f"Total date window of APPOINTMENTS is {diff_days} days. \n        Min date: {df_train['waiting_days'].min()} \n        Max date: {df_train['waiting_days'].max()}")


# Cool. I will explore the date columns and try get some insight about the No-show patterns <br>
# <br>
# Based on the quantiles and the range of days to the appointment, I will filter the data and get the range -1 to 100.
# 
# I think that -1 don't make any sense, but I think that it could be emergence appointments and zero is when the Patient Schedule in the same day.

# ## Ploting the waiting days by percent of Show and No-show Patients
# - As the 99 quantile of our data is 65, I will consider the data between -1 to 70 and show the ratio of the target to each schedule time difference

# In[ ]:


group_temp = df_train[(df_train['waiting_days'] < 70) & (df_train['waiting_days'] >= -1)].groupby(['waiting_days', 'No-show'])['PatientId'].count()  / df_train[(df_train['waiting_days'] < 70) & (df_train['waiting_days'] >= -1)].groupby(['waiting_days'])['PatientId'].count() 

# plt.figure(figsize=(14,6))

# sns.countplot(x='waiting_days', hue='No-show', data=df_train[(df_train['waiting_days'] < 10) & (df_train['waiting_days'] >= -1)]) 
fig = group_temp.unstack().iplot(kind='bar', barmode='stack', asFigure=True, title='Percent of Show and No-show Patients by Days to Appointment', xTitle='Days to Appointment', yTitle='Percent Show and No-Show') 

fig.layout.xaxis.type = 'category'
iplot(fig)


# I Think that -1 could be emergence. 

# ## Filtering and replacing the data 
# - Getting the values with range difference into -1 to 100

# In[ ]:


df_train = df_train[(df_train['waiting_days'] >= -1) & (df_train['waiting_days'] <=100)]


# 

# In[ ]:


df_train.groupby([df_train.ScheduledDay.dt.date, 'No-show' ])['PatientId'].count().unstack().fillna(0).iplot(kind='bar', barmode='stack', title='Appointments Dates and the distribution  of Show and No-show Patients', xTitle='Dates', yTitle='Count ' ) 


# Maybe will be interesting to get the data highest than march 2016

# In fact, the distribution is 

# many_appointments_patients = df_train.groupby(['PatientId'])['AppointmentID'].count().sort_values(ascending=False).head(10)
# df_train[df_train.PatientId.isin(many_appointments_patients.index)]['No-show'].value_counts(normalize=True).plot(kind='bar')

# # Start Modelling
# ## Preprocessing

# In[ ]:


df_train.Gender = df_train['Gender'].map({"F":0, "M":1})
df_train['No-show'] = df_train['No-show'].map({"No":0, "Yes":1})
df_train = dummies(df_train, categorical)

df_train.drop(['_weekdayName_AppointmentDay', 'AppointmentID', 'PatientId', 'age_bin', 'ScheduledDay', 'AppointmentDay', '_weekdayName_AppointmentDay', '_weekdayName_ScheduledDay'], axis=1, inplace=True) 


# In[ ]:


df_train.astype(float).corr()['No-show'].sort_values(ascending=False).head(10)


# ## Correlation matrix of all features

# In[ ]:


#Finallt, lets look the correlation of df_train
plt.figure(figsize=(20,15))
plt.title('Correlation of Features for Train Set')
print()
print()


# ## Seting X and y to train the model and spliting into validation set 

# In[ ]:


y_train = df_train['No-show']
X_train = df_train.drop('No-show', axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.25)


# In[ ]:


#Importing the auxiliar and preprocessing librarys 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import accuracy_score

#Models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, RandomTreesEmbedding


# ## Pipelining the models and ploting the cross validation results

# In[ ]:


clfs = []
seed = 3

clfs.append(("LogReg", Pipeline([("Scaler", StandardScaler()), ("LogReg", LogisticRegression())]))) 

clfs.append(("XGBClassifier", Pipeline([("Scaler", StandardScaler()), ("XGB", XGBClassifier())]))) 
clfs.append(("KNN", Pipeline([("Scaler", StandardScaler()), ("KNN", KNeighborsClassifier())]))) 

clfs.append(("DecisionTreeClassifier", Pipeline([("Scaler", StandardScaler()), ("DecisionTrees", DecisionTreeClassifier())]))) 

clfs.append(("RandomForestClassifier", Pipeline([("Scaler", StandardScaler()), ("RandomForest", RandomForestClassifier())]))) 

clfs.append(("GradientBoostingClassifier", Pipeline([("Scaler", StandardScaler()), ("GradientBoosting", GradientBoostingClassifier(max_features=15, n_estimators=600))]))) 

clfs.append(("RidgeClassifier", Pipeline([("Scaler", StandardScaler()), ("RidgeClassifier", RidgeClassifier())]))) 

clfs.append(("BaggingRidgeClassifier", Pipeline([("Scaler", StandardScaler()), ("BaggingClassifier", BaggingClassifier())]))) 

clfs.append(("ExtraTreesClassifier", Pipeline([("Scaler", StandardScaler()), ("ExtraTrees", ExtraTreesClassifier())]))) 

#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'
scoring = 'accuracy'
n_folds = 10

results, names  = [], [] 

for name, model  in clfs:
    kfold = KFold(n_splits=n_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=-1) 
    names.append(name)
    results.append(cv_results)    
    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)
    
# boxplot algorithm comparison
fig = plt.figure(figsize=(15,6))
fig.suptitle('Classifier Algorithm Comparison', fontsize=22)
ax = fig.add_subplot(111)
sns.boxplot(x=names, y=results)
ax.set_xticklabels(names)
ax.set_xlabel("Algorithmn", fontsize=20)
ax.set_ylabel("Accuracy of Models", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
print()


# - We can see that just KNN and DecisionTrees hasn't 79%+ of score... <br>
# - I think that 79% of accuracy using the 10 kfold is an excellent baseline. 

# ## Getting the model optimization of XGBclassifier to predict No-show Patients
# - Now we will use the hyperopt model optimization to automatize our search and get the optimal parameters

# In[ ]:


import scipy as sp 
from sklearn.model_selection import RandomizedSearchCV
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from functools import partial
from sklearn.metrics import confusion_matrix


# In[ ]:


from sklearn.model_selection import StratifiedKFold

def objective(params):
    params = { 'max_depth': int(params['max_depth']), 'gamma': "{:.3f}".format(params['gamma']), 'reg_alpha': "{:.3f}".format(params['reg_alpha']), 'learning_rate': "{:.3f}".format(params['learning_rate']), 'gamma': "{:.3f}".format(params['gamma']), 'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']), } 
    
    clf = XGBClassifier( n_estimators=600, n_jobs=-1, **params ) 

    score = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=StratifiedKFold()).mean()
    print("Accuracy {:.8f} params {}".format(-score, params))
    return -score

space = { 'max_depth': hp.quniform('max_depth', 2, 8, 1), 'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.4), 'reg_lambda': hp.uniform('reg_lambda', 0.7, 1.0), 'learning_rate': hp.uniform('learning_rate', 0.05, 0.2), 'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0), 'gamma': hp.uniform('gamma', 0.0, 0.5), } 

best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50) 


# In[ ]:


best['max_depth'] = int(best['max_depth'])

print("BEST PARAMS: ", best)


# ## Predicting No-show Patients with the best parameters we found in hyperopt

# In[ ]:


clf = XGBClassifier( n_estimators=5000, n_jobs=-1, **best ) 


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

pred = clf.predict(X_val)


# In[ ]:


print(f'Accuracy of our Classifier with best Hyper Parameeters: {round(accuracy_score(y_val, pred, normalize=True),4)}')


# In[ ]:


class_names = df_train['No-show'].unique()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """ This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`. """ 
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix: ")
    else:
        print('Confusion matrix, without normalization: ')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    print()
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),  xticklabels=classes, yticklabels=classes, title=title, ylabel='True label', xlabel='Predicted label') 

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor") 

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black") 
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_val, pred, classes=class_names, title='Confusion matrix, without normalization') 

# Plot normalized confusion matrix
plot_confusion_matrix(y_val, pred, classes=class_names, normalize=True, title='Normalized confusion matrix') 

print()


# ## I will keep this analysis and I also will build a classification model. 
# ## Stay Tuned and votes up the kernel =) 
