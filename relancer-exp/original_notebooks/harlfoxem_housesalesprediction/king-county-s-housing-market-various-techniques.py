#!/usr/bin/env python
# coding: utf-8

# ## House price prediction using different regression algorithms.
# 

# ### This notebook includes some analyses of the things which gretly affect the house's price.
# ### Then we have the machine learning part using various regression algorithms.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = 8, 10


# In[ ]:


df = pd.read_csv("../../../input/harlfoxem_housesalesprediction/kc_house_data.csv", parse_dates = ['date']) # The parse_date will change date column to readable format


# In[ ]:


df.head()


# In[ ]:


df.info()


# #### Adding month and year column to the dataset

# In[ ]:


df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year


# #### Now, I though of movng the price column to the end. This is achived by storing it in another variable, dropping the column and again adding it back to dataframe

# In[ ]:


price = df['price']
df.drop('price', inplace=True, axis = 1)
df['price'] = price


# In[ ]:


df.head(2)


# ---

# ## Comming to analyses first.

# #### The most important is date vs price to check at what part of the year prices increase or decrease

# In[ ]:


# Month vs price
sns.barplot(x = df['month'], y = df['price'], data = df)


# #### This is evident from the above graph that the best month to buy a house is February.

# ---

# In[ ]:


heatMap = df.corr()
f, ax = plt.subplots(figsize=(25,16))
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)



# ---

# In[ ]:


sns.countplot(x = df['month'],data = df)


# #### While the best month to buy is February but the most of the sale happens in May followed by July.

# ---

# In[ ]:


# Seperating the data by year.
filter2015 = df['year'] == 2015
filter2014 = df['year'] == 2014


# In[ ]:


freq2014 = df[filter2014]['price']/df[filter2014]['sqft_living']
freq2015 = df[filter2015]['price']/df[filter2015]['sqft_living']


# In[ ]:


plt.hist(x = freq2014, bins = 10, histtype = 'stepfilled')
plt.xlabel('price/sqft_living')


# In[ ]:


plt.hist(x = freq2015, bins = 10, histtype = 'stepfilled')
plt.xlabel('price/sqft_living')


# In[ ]:


price2015 = sum(df[filter2015]['price'])/len(df[filter2015]['price'])
price2014 = sum(df[filter2014]['price'])/len(df[filter2014]['price'])
print('The average cost in the year 2015 is: ',price2015)
print('The average cost in the year 2014 is: ',price2014)
print('The percentage increase is: ',((price2015-price2014)*100)/price2014,'%')


# ---

# ### Coming over to machine learning.

# #### We will drop some columns which are not really helpful to us.

# In[ ]:


df.drop(['date', 'id'], inplace = True, axis = 1)


# In[ ]:


#seperating the dataset into independent variables and dependent variable
x = df.iloc[:,:-1].values    # All the independent variables. 
y = df.iloc[:,20].values     # dependent variable 'price'


# In[ ]:


#Splitting the dataset into test set and train set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0,test_size=0.35)


# ### First one is Multiple linear regression

# In[ ]:


from sklearn.linear_model import LinearRegression
MLregressor = LinearRegression()
MLregressor.fit(x_train, y_train)
scoreML = MLregressor.score(x_test,y_test)


# ---
# 

# ### Decision Tree Regression

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
regDT = DecisionTreeRegressor(random_state = 0, criterion = 'mae',min_samples_split=18, min_samples_leaf=10)
regDT.fit(x_train, y_train)


# In[ ]:


scoreDT = regDT.score(x_test,y_test)


# ---

# ### Random Forest Regression

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
regRF = RandomForestRegressor(n_estimators=400, random_state = 0)
regRF.fit(x_train,y_train)


# In[ ]:


scoreRF = regRF.score(x_test,y_test)


# ---

# ### SVM Regression

# In[ ]:


from sklearn.svm import SVR
regSVR = SVR(kernel = 'sigmoid',degree=5)
regSVR.fit(x_test,y_test)


# In[ ]:


scoreSVR = regSVR.score(x_test,y_test)


# ---

# In[ ]:


Scores = pd.DataFrame({'Classifiers': ['Multiple Linear Regression', 'Decision Tree', 'Random Forest', 'SVM'],'Scores': [scoreML, scoreDT, scoreRF, scoreSVR]})


# In[ ]:


Scores


# ### I'm using Random Forest regressor as the output generator

# In[ ]:


pd.options.display.float_format = '${:,.2f}'.format  #To format the output


# In[ ]:


regRF.predict(x_test)  #predicts the prices


# In[ ]:


output = pd.DataFrame({'Actual Price':y_test,'Predicted price': regRF.predict(x_test)})


# In[ ]:


output.to_csv('output.csv', index=False, encoding='utf-8')


# In[ ]:





