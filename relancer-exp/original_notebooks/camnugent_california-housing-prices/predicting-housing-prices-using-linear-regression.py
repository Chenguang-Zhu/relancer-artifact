#!/usr/bin/env python
# coding: utf-8

# Predicting the housing prices of California City. This notebook uses simple Linear Regression model with proper visualizations and descriptions.

# ## Importing Libraries and Dataset

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


housing = pd.read_csv("../../../input/camnugent_california-housing-prices/housing.csv")
housing.head()


# In[ ]:


housing.info()


# In[ ]:


housing[housing['total_bedrooms'].isnull()]
housing.loc[290]


# ### Graph Visualizations for all given attributes

# In[ ]:


housing.hist(bins=50, figsize=(20,20))
print()


# ### Fillling all NAN values of Dataframe with Mean value

# In[ ]:


housing['total_bedrooms'][housing['total_bedrooms'].isnull()] = np.mean(housing['total_bedrooms'])
housing.loc[290]


# ### Calculating Average Rooms and Bedrooms

# In[ ]:


housing['avg_rooms'] = housing['total_rooms']/housing['households']
housing['avg_bedrooms'] = housing['total_bedrooms']/housing['households']
housing.head()


# ### Finding correlation between the predictors

# In[ ]:


housing.corr()


# ### Calculating population per household

# In[ ]:


housing['pop_household'] = housing['population']/housing['households']
housing[:10]


# In[ ]:


housing['NEAR BAY']=0
housing['INLAND']=0
housing['<1H OCEAN']=0
housing['ISLAND']=0
housing['NEAR OCEAN']=0
housing.head()


# ### Converting the ocean proximity data to one-hot vectors 

# In[ ]:


housing.loc[housing['ocean_proximity']=='NEAR BAY','NEAR BAY']=1
housing.loc[housing['ocean_proximity']=='INLAND','INLAND']=1
housing.loc[housing['ocean_proximity']=='<1H OCEAN','<1H OCEAN']=1
housing.loc[housing['ocean_proximity']=='ISLAND','ISLAND']=1
housing.loc[housing['ocean_proximity']=='NEAR OCEAN','NEAR OCEAN']=1
housing.head()


# ## Applying Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split


train_x = housing.drop(['total_rooms','total_bedrooms','households','ocean_proximity','median_house_value'],axis=1)
train_y = housing['median_house_value']

X,test_x,Y,test_y = train_test_split(train_x, train_y, test_size=0.2)


# In[ ]:


clf = LinearRegression()
clf.fit(np.array(X),Y)


# In[ ]:


import math

def roundup(x):
   return int(math.ceil(x / 100.0)) * 100 
pred = list(map(roundup,clf.predict(test_x)))

print(pred[:10])
test_y[:10]


# ### Calculating root mean squared error in the regression model

# In[ ]:


from sklearn.metrics import mean_squared_error

predictions = clf.predict(test_x)
mse = mean_squared_error(test_y, predictions)
rmse = np.sqrt(mse)
rmse


