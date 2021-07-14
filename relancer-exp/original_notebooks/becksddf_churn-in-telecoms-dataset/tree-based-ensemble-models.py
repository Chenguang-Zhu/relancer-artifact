#!/usr/bin/env python
# coding: utf-8

# # Predicting churning customers

# ## Exploratory Data Analysis
# 
#  (1) Preliminary analysis
#  
#  (2) Explore each individual variable
# 
#  (3) Explore pairwise relationship between variables
# 
#  (4) Explore each variable against the target variable

# ### (1) Preliminary analysis

# In[ ]:


import pandas as pd
df = pd.read_csv("../../../input/becksddf_churn-in-telecoms-dataset/bigml_59c28831336c6604c800002a.csv", sep=',')



# In[ ]:


df.head(5)


# In[ ]:


df.dtypes


# In[ ]:


list(df.columns)


# In[ ]:


df.shape


# In[ ]:


print(df['phone number'].nunique())
print(df['state'].nunique()) 


# In[ ]:


df.isna().sum()


# ### (2) Explore each individual variable: visualizing the distributions of some columns
# The column 'state' is discrete and of high cardinality. Therefore using one-hot encoding may result in very sparse features. This feature may be useful.</br>
# The column 'area code': why only three values? 'area code' is of int64 type but it may be supposed to be considered as categorical values.</br>
# The column 'phone number' is the IDs of the records. It is unique to each record. We will not use it.</br>
# The target variable 'churn' is imbalanced (almost 6:1). We may need to tackle this problem.</br></br>
# 
# This is a binary classification problem. Each row of the data corresponds to a unique user that belongs to a single class (True/False).

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


fig, axs = plt.subplots(3, 4, figsize=(20, 10))

df.state.value_counts().plot(kind='bar', ax=axs[0,0])
axs[0,0].set_title('state')
df.hist(column='account length', ax=axs[0,1])
axs[0,1].set_title('account length')
df['area code'].value_counts().plot(kind='bar', ax=axs[0,2])
axs[0,2].set_title('area code')
df['international plan'].value_counts().plot(kind='bar', ax=axs[0,3])
axs[0,3].set_title('international plan')

df['voice mail plan'].value_counts().plot(kind='bar', ax=axs[1,0])
axs[1,0].set_title('voice mail plan')
df.hist(column='number vmail messages', ax=axs[1,1])
axs[1,1].set_title('number vmail messages')

df.hist(column='customer service calls', ax=axs[2,0])
axs[2,0].set_title('customer service calls')
df.churn.value_counts().plot(kind='bar', ax=axs[2,1])
axs[2,1].set_title('churn')



# In[ ]:


churn_true = len(df[df['churn'] == True].index)
churn_false = len(df[df['churn'] == False].index)
print('Churn rate is: {}. \nchurn_false/churn_true = {}. churn_false - churn_true = {}. \nThe data is imbalanced.' .format(churn_true / (churn_true + churn_false), churn_false / churn_true, churn_false - churn_true)) 


# In[ ]:


cols = list(df.columns)
cols.remove('state')
cols.remove('area code')
cols.remove('phone number')
cols.remove('international plan')
cols.remove('voice mail plan')
cols.remove('churn')

# Define a set of columns to be removed. They are not to be used as features.
cols_to_remove = {'phone number', } # 'churn' not included

print(cols)
print()
print(cols_to_remove)


# ### (3) Explore pairwise relationship between variables (scatterplot matrix & correlation matrix)
# We can see that there are linear correlations between each of the following column pairs: 'total day minutes' and 'total day charge', 'total eve minutes' and 'total eve charge', 'total night minutes' and 'total night charge', and 'total intl minutes' and 'total intl charge'. </br>
# 
# We can therefore remove the four '*** charge' columns.

# In[ ]:


print()
plt.tight_layout()
print()


# Correlation matrix between each pair of features
# This confirms the above observations.

# In[ ]:


cols


# In[ ]:


import numpy as np

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.0)
plt.figure(figsize=(10,10))
print()

print()


# In[ ]:


cols_to_remove.update(['total day charge', 'total eve charge', 'total night charge', 'total intl charge'])
cols_to_remove


# ### (4) Explore each variable against the target variable
# The observation:</br>
# positive and negative classes ('churn') of data have different distributions in the the features, especially in 'total day minutes', 'international plan' and 'customer service call', etc.

# In[ ]:


df.groupby(['churn']).mean()


# In[ ]:


fig, axs = plt.subplots(3, 4, figsize=(20, 12))
df.groupby(['churn'])['account length'].plot(kind='kde', legend=True, ax=axs[0,0])
axs[0,0].set_title('account length')
df.groupby(['churn'])['number vmail messages'].plot(kind='kde', legend=True, ax=axs[0,1])
axs[0,1].set_title('number vmail messages')
df.groupby(['churn'])['total day minutes'].plot(kind='kde', legend=True, ax=axs[0,2])
axs[0,2].set_title('total day minutes')
df.groupby(['churn'])['total day calls'].plot(kind='kde', legend=True, ax=axs[0,3])
axs[0,3].set_title('total day calls')
df.groupby(['churn'])['total eve minutes'].plot(kind='kde', legend=True, ax=axs[1,0])
axs[1,0].set_title('total eve minutes')
df.groupby(['churn'])['total eve calls'].plot(kind='kde', legend=True, ax=axs[1,1])
axs[1,1].set_title('total eve calls')
df.groupby(['churn'])['total night minutes'].plot(kind='kde', legend=True, ax=axs[1,2])
axs[1,2].set_title('total night minutes')
df.groupby(['churn'])['total night calls'].plot(kind='kde', legend=True, ax=axs[1,3])
axs[1,3].set_title('total night calls')
df.groupby(['churn'])['total intl minutes'].plot(kind='kde', legend=True, ax=axs[2,0])
axs[2,0].set_title('total intl minutes')
df.groupby(['churn'])['total intl calls'].plot(kind='kde', legend=True, ax=axs[2,1])
axs[2,1].set_title('total intl calls')
df.groupby(['churn'])['customer service calls'].plot(kind='kde', legend=True, ax=axs[2,2])
axs[2,2].set_title('customer service calls')

# Because the classes are imbalanced, I think 'kde' is more preferred than 'hist' here.


# In[ ]:


fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.countplot(x='international plan', hue='churn', data=df, ax=axs[0])
# axs[0].set_title('international plan')
sns.countplot(x='voice mail plan', hue='churn', data=df, ax=axs[1])
# axs[1].set_title('voice mail plan')
sns.countplot(x='customer service calls', hue='churn', data=df, ax=axs[2])


# # Data preprocessing
# 
# This includes encoding categorical variables, etc.
# I did not do feature scaling since I'm going to use tree-based algorithms. 
# There is no null values.

# In[ ]:


cols_to_remove


# In[ ]:


df2 = df.drop(list(cols_to_remove), axis=1)
df2.head()


# Applying encoding to categorical variables

# In[ ]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# The following two are of multiple categories
# ohencoder = OneHotEncoder()

# TODO
# [Observations] 'state' and 'area code' columns are categorical and should be applied with one-hot encoding. 
# However due to 'state' is of high cardinality (51), this would negatively affect the prediction performance (low split gain)
# RF may be affected more than GBDT.
# 'area code' is of 3 categories. I applied one-hot encoding on it.


# The following three are of binary categories
label_encoder = LabelEncoder()
df2['international plan'] = label_encoder.fit_transform(df2['international plan'])
df2['voice mail plan'] = label_encoder.fit_transform(df2['voice mail plan'])
df2['churn'] = label_encoder.fit_transform(df2['churn'])


df2.head()


# In[ ]:


# one-hot encoding 'area code'
df2 = pd.get_dummies(df2, columns=['area code'], prefix='areacode', drop_first=True)
df2.head(10)


# # Churn Prediction

# ## Splitting the data for training and testing

# Select the features to be used, and then split the data into training dataset and test dataset. Make sure the proportion of different target values is consistent by using 'stratify=y'.

# In[ ]:


from sklearn.model_selection import train_test_split

X = df2.loc[:, [c for c in list(df2.columns) if c not in cols_to_remove | {'churn', 'state', 'area code'}]].values
y = df2.loc[:, 'churn'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


# In[ ]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# Because the data is imbalanced, I'll oversample the minor class (Churn=True) to make the data more balanced using SMOTE.

# In[ ]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=0, ratio=1.0)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)


# In[ ]:


X_train_balanced.shape, y_train_balanced.shape


# In[ ]:


import collections
collections.Counter(y_train_balanced)


# ## Training and evaluating the models 
# 
# I used tree-based algorithms.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=10, criterion='entropy')
rf.fit(X_train_balanced, y_train_balanced)
y_pred = rf.predict(X_test)

from sklearn.metrics import classification_report, f1_score, roc_auc_score
print(classification_report(y_test, y_pred))

metric_result = pd.DataFrame(data=[['RandomForestClassifier', f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred)]], columns=['algorithm', 'f1_score', 'roc_auc_score']) 

del y_pred


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=3)
ada = AdaBoostClassifier(base_estimator=tree, n_estimators=500, learning_rate=0.1, random_state=1)
ada.fit(X_train_balanced, y_train_balanced)
y_pred = ada.predict(X_test)

print(classification_report(y_test, y_pred))
metric_result.loc[1] = ['AdaBoostClassifier', f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred)]

del y_pred


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(loss='deviance', n_estimators=100, max_depth=4)
gbc.fit(X_train_balanced, y_train_balanced)
y_pred = gbc.predict(X_test)

print(classification_report(y_test, y_pred))
metric_result.loc[2] = ['GradientBoostingClassifier', f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred)]

del y_pred


# In[ ]:


metric_result


# In[ ]:


metric_result.plot(x='algorithm', kind='barh')


# After I tried one-hot encoding on the 'state' feature, I found that it seems Random Forest tends to be more affected by the encoding, in comparison to Gradient Boosting.

# # Next

# LightGBM, with Optimal Split for Categorical Features (https://lightgbm.readthedocs.io/en/latest/Features.html#optimal-split-for-categorical-features)
# 
# AdaCost: Misclassification Cost-sensitive Boosting

# In[ ]:




