#!/usr/bin/env python
# coding: utf-8

# ## Data Exploration

# In[ ]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# In[ ]:


rawData = pd.read_csv("../../../input/aljarah_xAPI-Edu-Data/xAPI-Edu-Data.csv")


# In[ ]:


rawData.tail().T


# In[ ]:


rawData.columns


# In[ ]:



rawData.info()


# In[ ]:


import numpy as np

np.where(pd.isnull(rawData))


# In[ ]:


rawData.describe()


# ###  Catagorical variables

# In[ ]:


cat_Vars = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID','Relation', 'GradeID', 'Topic', 'Semester', 'StudentAbsenceDays']


# In[ ]:


fig = plt.figure(figsize=(20, 30))
fig.subplots_adjust(hspace=.3, wspace=0.2)

for i in range(1,len(cat_Vars)+1,1):
    ax = fig.add_subplot(5, 2, i,)
    sns.countplot(rawData[cat_Vars[i-1]])
    ax.xaxis.label.set_size(20)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    total = float(len(rawData))
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,height + 5,'{:1.1f}%'.format(100 * height/total),ha="center")


# #### Based on the the above plots several observations can be made:
#     -Majority of students are male
#     -Most of the students are from either Kuwait or Jordan
#     -Majority of students are middleschool or lower

# ### Numerical Variables

# In[ ]:


num_Vars = [ 'raisedhands','VisITedResources', 'AnnouncementsView', 'Discussion', 'Class']

# fig = plt.figure(figsize=(10, 10))
# for i in range(1,len(num_Vars)+1,1):
#     ax = fig.add_subplot(2, 2, i)
#     sns.distplot(rawData[num_Vars[i-1]],kde= None)


# In[ ]:




# In general these features represent the level of student particapation in their respective courses. 
# Cleary the students with low grades tend to have lower level of class particapation.

# ## Lets try to figure out which feauters are most important (if any) for predicting students' academic perfomance.

# Lets predict student perfomance using scikit random forest classifier

# In[ ]:


target = rawData.pop('Class') # target

# Drop the features not relevant to the student perfomance
rawData.drop(rawData[['ParentAnsweringSurvey', 'ParentschoolSatisfaction']], axis=1, inplace=True)

X = pd.get_dummies(rawData) # get numeric dummy variables for categorical data


# In[ ]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
y = le.fit_transform(target) # encode target labels with a value


# In[ ]:


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.40) # split data set into train and set
X_train.shape, X_test.shape


# ### Cross-Validation to determine the best model parameter values

# In[ ]:


from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

param_grid ={'n_estimators': [50, 100, 200],'max_features': ['auto', 'sqrt', 'log2'],'min_samples_leaf' : [1, 5,10, 50],}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5, refit=True)
CV_rfc.fit(X_train, y_train)
CV_rfc.best_params_


# In[ ]:


#CV_rfc.grid_scores_


# ### Use best fit model parameters to make predictions on test set

# In[ ]:


from sklearn.metrics import accuracy_score

rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=100, oob_score = True)
rfc.fit(X_train, y_train)
pred = rfc.predict(X_test)
accuracy_score(y_test, pred)


# ### Plot feature importance 

# In[ ]:


feature_importance = pd.DataFrame(rfc.feature_importances_, index=X_train.columns, columns=["Importance"])
feature_importance.head(8)


# In[ ]:


feature_importance.sort('Importance', ascending=False, inplace=True)


# In[ ]:


fig = plt.figure(figsize=(25, 10))
ax = sns.barplot(feature_importance.index, feature_importance['Importance'])
fig.add_axes(ax)
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')[1]


# ### Quick test with xgboost classifier

# In[ ]:


import xgboost as xgb

xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test, label=y_test)


# In[ ]:


param = {}

# use softmax multi-class classification
param['objective'] = 'multi:softmax'

# set xgboost parameter values
param['eta'] = 0.1
param['max_depth'] = 9
param['silent'] = 1
param['nthread'] = 3
param['num_class'] = 3

num_round = 100
bst = xgb.train(param, xg_train, num_round)

# get prediction
pred = bst.predict( xg_test ).astype(int)
accuracy_score(y_test, pred)


# In[ ]:


xgb.plot_importance(bst, height=0.5)


# The features related to the student participation in the class are important in accurate prediction of the student
# performance in any given course. (This is specific to the data set and the model we are using in this analysis.) 
# This can be further tested by studying the variation in acuracy obtained by removing the important featues one by one from the model.

# In[ ]:





