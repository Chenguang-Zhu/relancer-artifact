#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15.0, 8.0)

from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
# Input data files are available in the "../../../input/camnugent_california-housing-prices/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../../../input/camnugent_california-housing-prices"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


housing_df = pd.read_csv("../../../input/camnugent_california-housing-prices/housing.csv")
housing_df.head()
housing_df = housing_df.drop('ocean_proximity',axis=1)
housing_df.head()


# In[ ]:


print('Shape :',housing_df.shape)
print('Features Data types : \n',housing_df.dtypes)
print('checking if any null values')
print(housing_df.isnull().sum())     


# In[ ]:


# Null values in total_bedrooms so we will drop them, Its best practice to replace any null values with mean/median.
housing_df = housing_df.dropna(axis=0)
housing_df.shape


# #### Our taget variable is median_house value, and so all the other columns are features. Now split the data into X and y variables where X considered as features and y target variable

# In[ ]:


X = housing_df.drop(['median_house_value'],axis=1)
y = housing_df['median_house_value']
print(X.shape,y.shape)


# In[ ]:


reg = linear_model.LinearRegression()


# *Split the data into train and test tests*

# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2)


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# In[ ]:


reg.fit(X_train,Y_train)


# In[ ]:


pred = reg.predict(X_test)


# In[ ]:


print('Predicted Value :',pred[0])
print('Actual Value :',Y_test.values[0])


# In[ ]:


res = pd.DataFrame({'Predicted':pred,'Actual':Y_test})
res = res.reset_index()
res = res.drop(['index'],axis=1)


# In[ ]:


plt.plot(res[:30])
plt.legend(['Actual','Predicted'])


# In[ ]:


res


# In[ ]:


#you can save the model and predict with single array values as below
# 'a' represent the list of features we have to predict the value of the house
a = [ -122.23,37.86,21.0,7099,1106.0,2401.0,1138.0,8.3014]
reg.predict(np.array(a).reshape([1,-1]))


