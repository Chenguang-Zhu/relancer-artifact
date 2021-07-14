#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 25, 10
from datetime import datetime
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import normalize
from sklearn.grid_search import GridSearchCV


# In[ ]:


df = pd.read_csv("../../../input/sudalairajkumar_indian-startup-funding/startup_funding.csv")
df = df.drop(['SNo','Remarks','SubVertical'],axis = 1)
df = df.dropna()
df = df.reset_index(drop=True)


# In[ ]:



#df['Date'] = ['' if df['Date'][dt].find('.') < 0 else '' for dt in range(0, len(df['Date']))]
for dt in range(0, len(df['Date'])):
    try:
        df['Date'][dt] = datetime.strptime(df['Date'][dt], '%d/%m/%Y').date()
       
    except:
        if dt > 0:
            df['Date'][dt] = df['Date'][dt-1]
        else:
            df['Date'][dt] = df['Date'][dt+1]
df.head()



# In[ ]:


df["month"] = [str(dt.month) for dt in df.Date]


# In[ ]:


for i in range(0,len(df.AmountInUSD)):
    df.AmountInUSD[i] = int(df.AmountInUSD[i].replace(",",""))

df['Amount_USD'] = list(map(int,df.AmountInUSD))


# In[ ]:


df = df.drop('AmountInUSD',axis = 1)

#df = df.groupby(['month','Date','CityLocation']).sum().reset_index()
df.info()


# In[ ]:


count = df['IndustryVertical'].value_counts()
count.head(10)
# plt.figure(figsize=(12,6))
# sns.barplot(count.index, count.values, alpha=0.8)
# plt.xticks(rotation='vertical')
# plt.xlabel('Investment Type', fontsize=12)
# plt.ylabel('Number of fundings made', fontsize=12)
# plt.title("Type of Investment made", fontsize=16)
# print()


# In[ ]:



count = df['InvestmentType'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(count.index, count.values, alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('Investment Type', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Type of Investment made", fontsize=16)
print()


# In[ ]:



count = df['month'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(count.index, count.values, alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('Month Wise Investment', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Type of Investment made", fontsize=16)
print()


# In[ ]:


# count = df['month'].value_counts()
# amount = df.Amount_USD
month_mean = df.groupby(['month']).sum().groupby(level=0).mean().reset_index()
print(month_mean)

#plt.bar(month_mean.index,month_mean.Amount_USD)
# plt.figure(figsize=(12,6))
sns.barplot(month_mean.index, month_mean.Amount_USD, alpha=0.8)
plt.xticks()
# plt.xlabel('Month Wise Investment', fontsize=12)
# plt.ylabel('Number of fundings made', fontsize=12)
# plt.title("Type of Investment made", fontsize=16)
# print()


# In[ ]:


count = df['CityLocation'].value_counts()
plt.figure(figsize=(15,6))
sns.barplot(count.index, count.values, alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('Investment Location', fontsize=25)
plt.ylabel('Number of fundings made', fontsize=25)
plt.title("Type of Investment made", fontsize=30)
print()


# In[ ]:


month_mean = df.groupby(['month']).sum().groupby(level=0).mean().reset_index()
plt.scatter(month_mean.index,month_mean.Amount_USD)
print()


# In[ ]:


# month_mean = df.groupby(['month']).sum().groupby(level=0).mean().reset_index()
# x = month_mean.index
# y =month_mean.Amount_USD
# plt.bar(x,y)
# print()
df= df.drop('Date',axis = 1)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df.head()


# In[ ]:


train,test= train_test_split(df,test_size=0.2,random_state =10)
train_x = train.drop(['Amount_USD'],axis = 1)
train_y = train['Amount_USD']
test_x = test.drop(['Amount_USD'],axis = 1)
test_y = test['Amount_USD']
#print(train_y)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le1, le2, le3, le4, le5,le6 = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder() 
le1.fit(df.InvestmentType)
le2.fit(df.InvestorsName)
le3.fit(df.IndustryVertical)
le4.fit(df.StartupName)
le5.fit(df.CityLocation)
le6.fit(df.month)
# le1.transform(df.InvestmentType)
# le2.transform(df.InvestorsName)
# le3.transform(df.IndustryVertical)
# le4.transform(df.StartupName)
# le5.transform(df.CityLocation)


# In[ ]:


train_df = pd.DataFrame( { 'InvestmentType': le1.transform(train_x.InvestmentType), 'InvestorsName': le2.transform(train_x.InvestorsName), 'IndustryVertical': le3.transform(train_x.IndustryVertical), 'StartupName': le4.transform(train_x.StartupName), 'CityLocation': le5.transform(train_x.CityLocation), 'month': le6.transform(train_x.month)  }) 
test_df=  (pd.DataFrame( {  'InvestmentType': le1.transform(test_x.InvestmentType), 'InvestorsName': le2.transform(test_x.InvestorsName), 'IndustryVertical': le3.transform(test_x.IndustryVertical), 'StartupName': le4.transform(test_x.StartupName), 'CityLocation': le5.transform(test_x.CityLocation), 'month': le6.transform(test_x.month) })) 
test_df.head()


# In[ ]:


clf = GradientBoostingRegressor(learning_rate =0.1,max_depth = 11,min_samples_split =100,min_samples_leaf =20,n_estimators =40, max_features =3,random_state =43) 
clf.fit(train_df,train_y)


# In[ ]:



from sklearn.metrics import mean_squared_error
pred = clf.predict(test_df)
RMSE= np.sqrt(mean_squared_error(test_y,pred))
RMSE 


# In[ ]:


# parameter1 = {'learning_rate':(0.1,0.05,0.03,0.01),'max_depth':(6,7, 8, 9, 10,11 ),'min_samples_split':(30,50,100,150,200),
#               'min_samples_leaf':(20,30,40,50,60,70,80),'n_estimators':(40,60,80,100,120),'max_features':(3,4)}
# gsearch1 = GridSearchCV(clf,parameter1)
# gsearch1.fit(train_df,train_y)
# gsearch1.grid_scores_, gsearch1.best_params_,gsearch1.best_score_


# In[ ]:


# parameter1 = {'learning_rate':(0.1,0.05),'max_depth':(6,7, 8, 9, 10,11 ),'min_samples_split':(50,80,100,120),
#               'min_samples_leaf':(15,20,25,30,35,40),'n_estimators':(40,60,80,100,120),'max_features':(3,4)}
# gsearch1 = GridSearchCV(clf,parameter1)
# gsearch1.fit(train_df,train_y)
# gsearch1.grid_scores_, gsearch1.best_params_,gsearch1.best_score_


# In[ ]:


# parameter1 = {'max_depth':(8, 9, 10,11,12 ),'min_samples_split':(50,80,100,120),
#               'min_samples_leaf':(5,10,15,20,25),'n_estimators':(80,90,100,110),'max_features':(2,3)}
# gsearch1 = GridSearchCV(clf,parameter1)
# gsearch1.fit(train_df,train_y)
# gsearch1.grid_scores_, gsearch1.best_params_,gsearch1.best_score_


# In[ ]:


clf1 = RandomForestRegressor(max_features =3,n_estimators = 30,min_samples_leaf =5 ,min_samples_split =50 ,max_depth = 6)
clf1.fit(train_df,train_y)


# In[ ]:


pred = clf1.predict(test_df)
RMSE = np.sqrt(mean_squared_error(test_y,pred))
RMSE 


# In[ ]:


# parameter2 = {'max_depth':(5,6,7,8 ),'min_samples_split':(50,80,100,120),
#               'min_samples_leaf':(5,10,15,20,25),'max_features':(2,3,4),'n_estimators':(20,30,40,50,100)}
# gsearch2 = GridSearchCV(clf1,parameter2)
# gsearch2.fit(train_df,train_y)
# gsearch2.grid_scores_, gsearch2.best_params_,gsearch2.best_score_


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
clf2 = LinearRegression()
clf2.fit(train_df,train_y)


# In[ ]:


pred = clf2.predict(test_df)
RMSE = np.sqrt(mean_squared_error(test_y,pred))
RMSE 


# In[ ]:


train_scale =normalize(train_df)
test_scale = normalize(test_df)
optimizer = SGDRegressor(n_iter=10000)
optimizer.fit(train_scale,train_y)


# In[ ]:





pred = optimizer.predict(test_scale)
optimizer.score(test_scale,test_y)
RMSE = np.sqrt(mean_squared_error(test_y,pred))
RMSE 


# In[ ]:





# In[ ]:




