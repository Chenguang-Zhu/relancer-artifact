#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from pandas import set_option


# In[ ]:


bicycle = pd.read_csv("../../../input/budincsevity_szeged-weather/weatherHistory.csv")


# In[ ]:


bicycle.head()


# In[ ]:


bicycle = bicycle.rename(columns={'Unnamed: 0' : 'Record Number'})
bicycle = bicycle.set_index('Date')
bicycle.drop('Day',axis=1,inplace=True)
bicycle.head(3)


# In[ ]:


bicycle.dtypes


# In[ ]:


temp = bicycle['Precipitation'].replace(['0.47 (S)'], '0.47')
temp = temp.replace(['T'], '0')
temp = temp.astype(float)


# In[ ]:


bicycle['Precipitation'] = temp
bicycle.head(4)


# In[ ]:


bicycle.Precipitation.value_counts()


# In[ ]:


set_option('precision',1)
bicycle.describe()


# In[ ]:


set_option('precision',2)
x = bicycle.corr()
x


# In[ ]:


plt.subplots(figsize=(10,6))
print()
print()


# In[ ]:


temp = bicycle['High Temp (°F)'].describe()
temp


# In[ ]:


bicycle['High Temp <25%'] = (bicycle['High Temp (°F)'] <= temp['25%']).astype(int)
bicycle['High Temp >25&<50%'] = ((bicycle['High Temp (°F)'] > temp['25%']) & (bicycle['High Temp (°F)'] <= temp['50%'])).astype(int)
bicycle['High Temp >50&<75%'] = ((bicycle['High Temp (°F)'] > temp['50%']) & (bicycle['High Temp (°F)'] <= temp['75%'])).astype(int)


# In[ ]:


bicycle.head()


# In[ ]:


temp = bicycle['Low Temp (°F)'].describe()
temp


# In[ ]:


bicycle['Low Temp <25%'] = (bicycle['Low Temp (°F)'] <= temp['25%']).astype(int)
bicycle['Low Temp >25&<50%'] = ((bicycle['Low Temp (°F)'] > temp['25%']) & (bicycle['Low Temp (°F)'] <= temp['50%'])).astype(int)
bicycle['Low Temp >50&<75%'] = ((bicycle['Low Temp (°F)'] > temp['50%']) & (bicycle['Low Temp (°F)'] <= temp['75%'])).astype(int)


# In[ ]:


bicycle.head(2)


# In[ ]:


temp = bicycle['Precipitation'].describe()
temp


# In[ ]:


bicycle.Precipitation.value_counts().sort_index()


# In[ ]:


bicycle['Precp >75%'] = (bicycle['Precipitation'] >= temp['75%']).astype(int)
bicycle['Precp min'] = (bicycle['Precipitation'] == 0).astype(int)
bicycle['Precp 1'] = (bicycle['Precipitation'] < 0.09).astype(int)
bicycle['Precp 2'] = (bicycle['Precipitation'] <= 0.15).astype(int)
bicycle['Precp 3'] = (bicycle['Precipitation'] <= 0.20).astype(int)


# In[ ]:


print()
#print()


# In[ ]:


X = bicycle.drop(['Total','Record Number','High Temp (°F)','Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge','Queensboro Bridge'],axis=1)
y = bicycle['Total']


# In[ ]:


lin = LinearRegression()


# In[ ]:


Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,random_state=1)
print(Xtrain.shape)
print(Xtest.shape)
print(ytrain.shape)
print(ytest.shape)


# In[ ]:


lin.fit(Xtrain,ytrain)
print(lin.intercept_)
lin.coef_


# In[ ]:


y_pred = lin.predict(Xtest)
np.sqrt(metrics.mean_squared_error(ytest,y_pred))


# In[ ]:


df = pd.DataFrame({})
df = pd.concat([Xtest,ytest],axis=1)
df['Predicted'] = np.round(y_pred,2)
df['ERROR'] = df['Total'] - df['Predicted']


# In[ ]:


df.head(2)


# In[ ]:


df['ERROR'].describe()


# In[ ]:





# In[ ]:




