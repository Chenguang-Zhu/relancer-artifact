#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../../../input/wenruliu_adult-income-dataset/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../../../input/wenruliu_adult-income-dataset"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Import the dataset
data = pd.read_csv("../../../input/wenruliu_adult-income-dataset/adult.csv")
data.head()



# We modify column names 

# In[ ]:



my_columns = ['Age', 'Workclass', 'Fnlwgt', 'Education', 'Education_num', 'Martial_status', 'Occupation', 'Relationship', 'Race', 'Gender', 'Capital_gain','Capital_loss', 'hours_per_week', 'Native_country','Target']
data.columns = my_columns
data.head()


# In[ ]:


# Check the data type of each column and number of rows and columns present in the dataset
print(data.info())
print(data.shape)


# We can see there are 48842 rows and 15 columns. Some of the variables are continous and while others are categorical.

# In[ ]:


# Check whether there are any missing values
col_names = data.columns
num_data = data.shape[0]
for c in col_names:
    num_non = data[c].isin(["?"]).sum()
    if num_non > 0:
        print (c)
        print (num_non)
        print ("{0:.2f}%".format(float(num_non) / num_data * 100))
        print ("\n")


# In[ ]:


new_data = data
new_data.head()


# In[ ]:


from scipy.stats import mode
var = new_data.Workclass.mode()[0]
new_data.loc[new_data['Workclass'] == '?', 'Workclass'] = var

var1 = new_data.Occupation.mode()[0]
new_data.loc[new_data['Occupation'] == '?', 'Occupation'] = var1

var2 = new_data.Native_country.mode()[0]
new_data.loc[new_data['Native_country'] == '?', 'Native_country'] = var2
new_data.head()


# Now all the missing values are replaced by mode of the column

# EDA

# In[ ]:


my_tab = pd.crosstab(index = data.Education,   columns= data.Target)       

my_tab.plot.bar()


# In[ ]:


my_tab = pd.crosstab(index = data.Race,   columns= data.Target)       

my_tab.plot.bar()


# In[ ]:


my_tab = pd.crosstab(index = data.Martial_status,   columns= data.Target)       

my_tab.plot.bar()


# In[ ]:


my_tab = pd.crosstab(index = data.Native_country,   columns= data.Target)       

my_tab.plot.bar()


# 

# **Feature Encoding**

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
Workclass_cat = le.fit_transform(new_data.Workclass)
Education_cat = le.fit_transform(new_data.Education)
Marital_cat   = le.fit_transform(new_data.Martial_status)
Occupation_cat = le.fit_transform(new_data.Occupation)
Relationship_cat = le.fit_transform(new_data.Relationship)
Race_cat = le.fit_transform(new_data.Race)
Sex_cat = le.fit_transform(new_data.Gender)
Native_country_cat = le.fit_transform(new_data.Native_country)
Target_cat = le.fit_transform(new_data.Target)



# In[ ]:


# Initialize the transformed values
new_data['Workclass'] = Workclass_cat
new_data['Education'] = Education_cat
new_data['Martial_status'] = Marital_cat
new_data['Occupation'] = Occupation_cat
new_data['Race'] = Race_cat
new_data['Gender'] = Sex_cat
new_data['Native_country'] = Native_country_cat


# In[ ]:


#dummy_fields = ['Workclass', 'Education', 'Martial_status', 'Occupation', 'Race','Sex', 'Native_country']
new_data.head()


# In[ ]:


# split the target and predictor variables
X = new_data.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12,13]]
Y = new_data.iloc[:,14]
X.head()


# In[ ]:


# Split the dataset into train and test dataset

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 5)


# In[ ]:


# Pipe line for multimodeling 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score 

lr = LogisticRegression()
Lr_model = lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)
acc_score_lr = accuracy_score(Y_test, Y_pred) 
print(acc_score_lr)


# In[ ]:


DTC = DecisionTreeClassifier()
DTC_model = DTC.fit(X_train, Y_train)
Y_pred = DTC.predict(X_test)
acc_score_DTC = accuracy_score(Y_test, Y_pred)
print(acc_score_DTC)


# In[ ]:


RF = RandomForestClassifier(n_estimators=1000, n_jobs= 2, random_state= 5)
RF_model = RF.fit(X_train, Y_train)
Y_pred = RF.predict(X_test)
acc_score_RF = accuracy_score(Y_test, Y_pred)
print(acc_score_RF)


# In[ ]:


SVM = SVC()
SVM_model = SVM.fit(X_train, Y_train)
Y_pred = SVM.predict(X_test)
acc_score_SVM = accuracy_score(Y_test, Y_pred)
print(acc_score_SVM)


# In[ ]:




