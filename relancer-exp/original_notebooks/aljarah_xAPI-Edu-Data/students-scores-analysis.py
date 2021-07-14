#!/usr/bin/env python
# coding: utf-8

# #Step by step process of analyzing data and finding the best predictor.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../../../input/aljarah_xAPI-Edu-Data/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../../../input/aljarah_xAPI-Edu-Data"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# First lets import everything

# In[ ]:


import seaborn as sns
sns.set(style='white')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error


# Reading the data and dropping place of birth due to similarity. Also looking at mean and std of the data.

# In[ ]:


df = pd.read_csv("../../../input/aljarah_xAPI-Edu-Data/xAPI-Edu-Data.csv")
#print df.shape

df = df.drop('PlaceofBirth',1)
#print df.head(5)

print (df.describe())


# Lets create count plots for better visualization of data. 

# In[ ]:


ls = ['gender','Relation','Topic','SectionID','GradeID','NationalITy','Class','StageID','Semester','ParentAnsweringSurvey','ParentschoolSatisfaction','StudentAbsenceDays']

for i in ls:
    g = sns.factorplot(i,data=df,kind='count',size=5,aspect=1.5)

print (df.shape)


# We can observe disproportionate difference in peaks of attributes such as nationality, GradeID.

# **Now let preprocess the data. First we do One hot Encoding to deal with categorical data .Then we split the data in train and test and also target and train. Finally we apply standard scaling to the data.**

# In[ ]:


#preprocessing

target = df.pop('Class')

X = pd.get_dummies(df)

le = LabelEncoder()
y = le.fit_transform(target)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
ss = StandardScaler()

X_train_std = ss.fit_transform(X_train)
X_test_std = ss.fit_transform(X_test)


# Now due to one hot encoding we significantly increase the number of attributes. In order to reduce them I used Random forest classifier which shows the importances of attributes with respect to the target.

# In[ ]:


#dimensionality_reduction

feat_labels = X.columns[:58]
forest = RandomForestClassifier(n_estimators=1000,random_state=0,n_jobs=-1)
forest.fit(X_train,y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))
h = sns.barplot(importances[indices],feat_labels[indices])


# As we observe first 6 attributes contribute significantly in the plot as compared to others. So we can safely remove others.

# **Now we remove unnecessary dimensions or attributes.** 
# Please someone tell me a more efficient way to do this.

# In[ ]:


#removing dimensions

X_train_new = X_train
X_test_new = X_test

ls = ['VisITedResources','raisedhands','AnnouncementsView','StudentAbsenceDays_Above-7','StudentAbsenceDays_Under-7','Discussion']

#Please someone tell me a more efficient way

for i in X_train.columns:
    if i in ls:
        pass
    else:
        X_train_new.drop(i , axis=1, inplace=True)
        
for i in X_test.columns:
    if i in ls:
        pass
    else:
        X_test_new.drop(i , axis=1, inplace=True)
        


# After this our data is ready to be spot checked by different algorithms.

# In[ ]:


#spot checking algorithms

models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X_train_new, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# We can observe that the best result is shown by Lasso but this data is not yet scaled.
# The scoring I have used is neg_mean_squared_error (0 means perfect)

# Now lets standardize the data and plot a boxplot for better visual on the errors.

# In[ ]:


# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN',ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X_train_new, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
  
fig = plt.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
print()


# We can see that although scaledLR does better, it has more variance. ScaledLasso and ScaledElasticNet have good score as well as less variance

# **Thus we tune Lasso and see if we can further improve the result.**

# In[ ]:


#lasso algorithm tuning
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
#the values chosen are done bby hit and trial during experimentations.
k_values = np.array([.1,.11,.12,.13,.14,.15,.16,.09,.08,.07,.06,.05,.04])
param_grid = dict(alpha=k_values)
model = Lasso()
kfold = KFold(n_splits=10, random_state=7)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
grid_result = grid.fit(rescaledX, y_train)


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# So the best value was obtained at alpha = 0.04 and other values are close as well

# **Lets use ensembles to see if we can improve the accuracy and also plot the results.**

# In[ ]:


#using ensembles
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB',AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM',GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF',RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET',ExtraTreesRegressor())])))
results = []
names = []
for name, model in ensembles:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
print()


# Scaled AdaBoost seems to do better than every other with less variance.

# **Now further tuning Adaboost**

# In[ ]:


# Tune scaled AdaboostRegressor
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=np.array([50,100,150,200,250,300,350,400]))
model = AdaBoostRegressor(random_state=7)
kfold = KFold(n_splits=10, random_state=7)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
grid_result = grid.fit(rescaledX, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# Result does not vary at all with n estimators so we can use any.

# **Finally lets prepare the model and see the final predictions**

# In[ ]:


# prepare the model by Adaboost
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = AdaBoostRegressor(random_state=7, n_estimators=400)
model.fit(rescaledX, y_train)


# In[ ]:


#prepare the model by LASSO
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model_l = Lasso(random_state=7, alpha=0.04)
model_l.fit(rescaledX, y_train)


# In[ ]:


# transform the validation dataset by adaboost
rescaledValidationX = scaler.transform(X_test)
predictions = model.predict(rescaledValidationX)
print(mean_squared_error(y_test, predictions))


# In[ ]:


# transform the validation dataset by lasso
rescaledValidationX = scaler.transform(X_test)
predictions = model_l.predict(rescaledValidationX)
print(mean_squared_error(y_test, predictions))


# #So we obtain somewhat 0.66 final error by adaboost and 0.72 by Lasso and during validation we got 0.61. This indicates little bit of overfitting in Adaboost but more in Lasso.

