#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../../../input/russellyates88_suicide-rates-overview-1985-to-2016/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk("../../../input/russellyates88_suicide-rates-overview-1985-to-2016"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import warnings
warnings.filterwarnings('ignore')

# Importing matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn import preprocessing
from sklearn import utils


# In[ ]:


file_path = "../../../input/russellyates88_suicide-rates-overview-1985-to-2016/master.csv"
# import the data
s_master = pd.read_csv(file_path)
s_master.head()


# In[ ]:


print(s_master.shape)
print(s_master.describe())


# In[ ]:


# Unique values in each column
s_master.nunique()


# In[ ]:


# % of null values in each column
round(100*(s_master.isnull().sum()/len(s_master.index)), 2)


# Human Development index is nearly 70% null. Since it is not correct to replace null with zeros because we are unsure of data we can drop this column going ahead. Also since we alrady have no of suicides we can skip suicides/100k pop

# In[ ]:


# Dropping redundant fields
s_master = s_master.drop(["HDI for year","country-year","suicides/100k pop"],axis=1)


# In[ ]:


# Renaming columns with proper convention
s_master = s_master.rename(columns={" gdp_for_year ($) ":"gdp_for_year_usd","gdp_per_capita ($)":"gdp_per_capita_usd"})
s_master.head()


# In[ ]:


# Normalizing the gdp_for_year_usd data
s_master["gdp_for_year_usd"] = s_master["gdp_for_year_usd"].str.replace(',','').astype('int64')


# In[ ]:


# Observe Unique Values
print(s_master["country"].unique())
print(s_master["year"].unique())
print(s_master["age"].unique())
print(s_master["generation"].unique())


# In[ ]:


# Let's see the correlation matrix 
plt.figure(figsize = (20,15))        # Size of the figure
print()


# Above plot shows correlation between continuous features in the dataset

# In[ ]:


# Let's see the Pair plot
plt.figure(figsize = (20,10))        # Size of the figure
print()


# ## Visualizing the Data

# #### Plotting Categorical Variables

# In[ ]:


def cfuncPlot(var,df):
    plt.figure(figsize=(20, 10))
    plt.subplot(2,2,1)
    sns.countplot(x=var,  data=df, order = df[var].value_counts(normalize=True, sort=True).index)
    plt.subplot(2,2,2)
    ax = (df[var].value_counts(normalize=True, sort=True)*100).plot.pie(autopct='%1.1f%%')
    print()


# In[ ]:


cfuncPlot("sex",s_master)


# In[ ]:


cfuncPlot("age",s_master)


# In[ ]:


cfuncPlot("generation",s_master)


# ### Bivariate Analysis
# 
# Total Suicides Count per country

# In[ ]:


# Total No of Suicides based on country
sui_no = s_master.groupby(['country']).suicides_no.sum()
sui_no1 = pd.DataFrame(sui_no)
sui_no1.reset_index(inplace=True)
plt.figure(figsize=(10,25))
ax = sns.barplot(x="suicides_no", y="country", data=sui_no1.sort_values(by=['suicides_no'],ascending=False), orient = "h")
plt.title('Counts by country')
print()

sui_no10 = sui_no1.sort_values(by=['suicides_no'],ascending=False).head(4)

# Top 4 countries with MAX Suicides
top_4 = list(sui_no10['country'])
print("The top 4 countries with highest number of Suicides are:")
print(top_4)


# In[ ]:


# GDP per capita based on country
plt.figure(figsize=(10,25))
ax = sns.barplot(x="gdp_per_capita_usd", y="country", data=s_master, orient = "h")
plt.title('GDP Per Capita for each County')
print()


# ###### Total suicides count by year

# In[ ]:


sui_no_yr = s_master.groupby(['year']).suicides_no.sum()
sui_no_yr1 = pd.DataFrame(sui_no_yr)
sui_no_yr1.reset_index(inplace=True) 
plt.figure(figsize=(10,25))
ax = sns.barplot(x="suicides_no", y="year", data=sui_no_yr1, orient = "h")
plt.title('Counts by Year')
print()


# ###### Suicide count distributed between sex

# In[ ]:


def sex_plot(df):    
    sui_no_sex = df.groupby(['sex']).suicides_no.sum()
    sui_no_sex = pd.DataFrame(sui_no_sex) 
    sui_no_sex.index = sui_no_sex.index.set_names(['sex'])
    sui_no_sex.reset_index(inplace=True)
    plt.figure(figsize=(25,20))
    plt.subplot(2,2,1)
    sns.barplot(x="sex", y="suicides_no", data=sui_no_sex)

    plt.subplot(2,2,2)
    explode = (0.05,0.05)
    ax = (sui_no_sex["suicides_no"]).plot.pie(autopct='%1.1f%%', labels=sui_no_sex['sex'].values,shadow=True, startangle=0,explode=explode)
    print()
    
sex_plot(s_master)   


# ###### Suicide count distributed between age group

# In[ ]:


def age_grp_plot(df):    
    sui_no_age = df.groupby(['age']).suicides_no.sum()
    sui_no_age = pd.DataFrame(sui_no_age) 
    sui_no_age.index = sui_no_age.index.set_names(['age'])
    sui_no_age.reset_index(inplace=True)
    plt.figure(figsize=(25,20))
    plt.subplot(2,2,1)
    sns.barplot(y="age", x="suicides_no", data=sui_no_age,orient = "h")
    plt.subplot(2,2,2)
    explode = (0.05,0.05,0.05,0.2,0.05,0.05)
    ax = (sui_no_age["suicides_no"]).plot.pie(autopct='%1.1f%%', labels=sui_no_age['age'].values,shadow=True, startangle=0,explode=explode)
    print()
    
age_grp_plot(s_master)    


# ###### Suicides distributed between generation

# In[ ]:


def gen_plot(df):    
    sui_no_gen = df.groupby(['generation']).suicides_no.sum()
    sui_no_gen = pd.DataFrame(sui_no_gen) 
    sui_no_gen.index = sui_no_gen.index.set_names(['generation'])
    sui_no_gen.reset_index(inplace=True)
    plt.figure(figsize=(25,20))
    plt.subplot(2,2,1)
    sns.barplot(y="generation", x="suicides_no", data=sui_no_gen,orient = "h")
    plt.subplot(2,2,2)
    explode = (0.05,0.05,0.05,0.2,0.05,0.05)
    ax = (sui_no_gen["suicides_no"]).plot.pie(autopct='%1.1f%%', labels=sui_no_gen['generation'].values,shadow=True, startangle=0,explode=explode)
    print()
    
gen_plot(s_master)


# ### Multivariate Analysis
# 
# Total Suicide count by sex and age group

# In[ ]:


def mv_1(df):    
    sui_no_sex = df.groupby(['sex','age']).suicides_no.sum()
    sui_no_sex = pd.DataFrame(sui_no_sex) #.reset_index(inplace=True)
    sui_no_sex.index = sui_no_sex.index.set_names(['sex', 'age'])
    sui_no_sex.reset_index(inplace=True)
    #print(sui_no_sex.head())
    plt.figure(figsize=(10,10))
    sns.barplot(x='sex', y='suicides_no', hue='age', data=sui_no_sex)
    
mv_1(s_master)    


# It can be clearly seen that in males the age group between 35-54 has the highest suicide rate.
# In Females also we observe a similar pattern but number are considerably less
# 
# #### Lets compare Generations now

# In[ ]:


def mv_2(df):    
    sui_no_sex = df.groupby(['sex','generation']).suicides_no.sum()
    sui_no_sex = pd.DataFrame(sui_no_sex) #.reset_index(inplace=True)
    sui_no_sex.index = sui_no_sex.index.set_names(['sex', 'generation'])
    sui_no_sex.reset_index(inplace=True)
    plt.figure(figsize=(10,10))
    sns.barplot(x='sex', y='suicides_no', hue='generation', data=sui_no_sex)
    
mv_2(s_master)  


# Among men boomers have the highest no of suicides. Refer https://en.wikipedia.org/wiki/Generation for more clarity on Generation
# 
# #### Lets do an yearly comparison of suicides between males and females

# In[ ]:


sui_no_yr = s_master.groupby(['year',"sex"]).suicides_no.sum()
sui_no_yr1 = pd.DataFrame(sui_no_yr)
sui_no_yr1.reset_index(inplace=True) 
plt.figure(figsize=(10,25))
ax = sns.barplot(x="suicides_no", y="year", hue='sex', data=sui_no_yr1, orient = "h")
plt.title('Counts by Year')
print()


# #### Lets do some country based analysis for top 4 suicide count countries

# In[ ]:


print("The top 4 countries with highest number of Suicides are:",top_4)


# In[ ]:


top4_df = s_master[s_master['country'].isin(top_4)]
top4_df.head(-1)


# ### Lets Do for Russia

# In[ ]:


df_Russia = top4_df[top4_df['country'].isin(["Russian Federation"])]
df_Russia.head()


# In[ ]:


sex_plot(df_Russia)
gen_plot(df_Russia)
mv_1(df_Russia)
mv_2(df_Russia)


# #### Trends in Russia seems to be inline with world. Rather I would say Russian trends implacted that of world
# 
# ### Lets try for United States

# In[ ]:


df_USA = top4_df[top4_df['country'].isin(["United States"])]
df_USA.head()


# In[ ]:


# Lets see the relations
sex_plot(df_USA)
age_grp_plot(df_USA)
gen_plot(df_USA)
mv_1(df_USA)
mv_2(df_USA)


# USA also shows similar trends

# ### Standardizing the Data

# In[ ]:


s_master.generation.replace(['Boomers', 'Generation X', 'Generation Z', 'G.I. Generation', 'Millenials', 'Silent'],['0', '1', '2', '3', '4', '5'], inplace=True)

s_master.sex.replace(['male', 'female'], ['0', '1'], inplace=True)

def means(arr):
    return str(np.array(arr).mean())
s_master.age.replace(['15-24 years', '25-34 years', '35-54 years', '5-14 years', '55-74 years', '75+ years'],[means([15, 24]), means([25, 34]), means([35, 54]),means([5, 14]), means([55, 74]), means([75])], inplace=True)
s_master = s_master.drop(['country'],axis=1)


# In[ ]:


# import standard scalar
from sklearn.preprocessing import StandardScaler

# extract columns
cols = s_master.columns

# apply standard scalar
scaler = StandardScaler()

# produce scaled features
s_master = scaler.fit_transform(s_master)

# convert to data frame
s_master = pd.DataFrame(s_master, columns=cols)
s_master.head()


# ## Model Implementation
# 
# We are using Random forest regressor for this part

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split


# In[ ]:


sucide_df = s_master.drop(["suicides_no"],axis=1)
y = s_master['suicides_no']


# In[ ]:


# divide the dataset into the train and test sections, keeping test size of30% 
train_x, test_x, train_y, test_y = train_test_split(sucide_df,y, test_size = 0.3)
print(train_x.shape, train_y.shape)
print(test_y.shape, test_x.shape)
print(train_x.head())


# In[ ]:


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_x, train_y)
pred_y = model.predict(test_x)


# In[ ]:


plt.figure(figsize=(10, 5))
plt.scatter(test_y, pred_y, s=20)
plt.title('Predicted vs. Actual')
plt.xlabel('Actual Suicides')
plt.ylabel('Predicted Suicides')

plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)])
plt.tight_layout()


# In[ ]:


# Evaluating the Algorithm
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, pred_y))  
print('Mean Squared Error:', metrics.mean_squared_error(test_y, pred_y))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, pred_y)))
print('R2 Score:', metrics.r2_score(test_y, pred_y))


# In[ ]:





