#!/usr/bin/env python
# coding: utf-8

# # Churn in Telecom's dataset 

# ## Index 

# * [Libraries](#Libraries) 
# * [Loading Data](#Loading-Data)
# * [Data Description](#Data-Description)
# * [Checking null values and value_counts](#Checking-null-values-and-value_counts)
# * [Data Visualization](#Data-Visualization)
# 	* [Churn Plot](#Churn-Plot)
#     * [Area code and Churn plot](#Area-code-and-Churn-plot)
#     * [State and Churn plot](#State-and-Churn-plot)
#     * [International plan and Churn](#International-plan-and-Churn)
#     * [Voice mail plan and Churn plot](#Voice-mail-plan-and-Churn-plot)
#     * [Customer service calls and Churn plot](#Customer-service-calls-and-Churn-plot)
# * [Label encoding](#Label-encoding)
# * [Model Building](#Model-Building)
# * [Feature importance using Random Forest](#Feature-importance-using-Random-Forest)
# * [Train test split](#Train-test-split)
# * [Random forest](#Random-forest)
# * [Decision tree](#Decision-tree)
# * [One-Hot Encoding](#One-Hot-Encoding)
# * [Logistic Regression](#Logistic-Regression)
# * [Saving to pkl file](#Saving-to-pkl-file)
# * [Load Pkl files](#Load-Pkl-files)
# * [All Classification Algorithm](#All-Classification-Algorithm)
#     * LogisticRegression
#     * XGBClassifier
#     * MultinomialNB
#     * AdaBoostClassifier
#     * KNeighborsClassifier
#     * GradientBoostingClassifier
#     * ExtraTreesClassifier
#     * DecisionTreeClassifier 

# ### Libraries

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score,roc_auc_score


# ### Loading Data

# In[ ]:


data=pd.read_csv("../../../input/becksddf_churn-in-telecoms-dataset/bigml_59c28831336c6604c800002a.csv")


# ### Data Description

# In[ ]:


print(data.shape)
print(data.columns)
data.head()


# In[ ]:


data.describe()


# In[ ]:


data.dtypes


# ### Checking null values and value_counts

# In[ ]:


data.isnull().sum()


# In[ ]:


data.drop(["phone number"],axis=1,inplace=True)


# In[ ]:


data["account length"].value_counts().head(20)


# In[ ]:


for i in data.columns:
    if data[i].dtype == "object":
        print(data[i].value_counts())


# ## Data Visualization

# ### Churn Plot

# In[ ]:


ax=sns.countplot(x="churn",data=data)
for p in ax.patches:
        ax.annotate('{:.1f}%'.format( (p.get_height()/data.shape[0])*100 ), (p.get_x()+0.3, p.get_height()))


# ### Area code and Churn plot

# In[ ]:


# data.groupby(["area code","churn"]).size()


# In[ ]:


ac=data.groupby(["area code", "churn"]).size().unstack().plot(kind='bar', stacked=False,figsize=(6,5))
for i in ac.patches:
    ac.text(i.get_x()+0.05, i.get_height()+20,str(i.get_height()))


# ###  State and Churn plot

# In[ ]:


# data["state"].value_counts()


# In[ ]:


st=data.groupby(["state", "churn"]).size().unstack().plot(kind='bar',stacked=True,figsize=(15,5))
# for i in st.patches:
#     st.text(i.get_x(), i.get_height(),str(i.get_height()))


# In[ ]:


# cols=['state','area code',
#  'international plan',
#  'voice mail plan',
#  'number vmail messages','customer service calls',]


# In[ ]:


# plt.plot([1,6])
# for i in range(len(cols)):
#     plt.subplot(i+1,1,1)
#     a=data.groupby([cols[i], "churn"]).size().unstack().plot(kind='bar', stacked=False,figsize=(6,5))
#     for i in a.patches:
#         a.text(i.get_x(), i.get_height(),str(i.get_height()))
#     print()


# ### International plan and Churn

# In[ ]:


ip=data.groupby(["international plan", "churn"]).size().unstack().plot(kind='bar', stacked=False,figsize=(6,5))
for i in ip.patches:
    ip.text(i.get_x()+0.05, i.get_height()+20,str(i.get_height()))


# ### Voice mail plan and Churn plot

# In[ ]:


vp=data.groupby(["voice mail plan", "churn"]).size().unstack().plot(kind='bar', stacked=True,figsize=(6,5))
# for i in vp.patches:
#     vp.text(i.get_x()+0.05, i.get_height()+20,str(i.get_height()))


# ### Customer service calls and Churn plot

# In[ ]:


cs=data.groupby(["customer service calls", "churn"]).size().unstack().plot(kind='bar', stacked=False,figsize=(12,6))
for i in cs.patches:
    cs.text(i.get_x()+0.05, i.get_height()+20,int(i.get_height()))


# In[ ]:


cate = [key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['bool', 'object']]


# ### Label encoding 

# In[ ]:


le = preprocessing.LabelEncoder()
for i in cate:
    le.fit(data[i])
    data[i] = le.transform(data[i])


# In[ ]:


data.head()


# ## Model Building

# ### Feature importance using Random Forest 

# In[ ]:


y=data["churn"]
x=data.drop(["churn"],axis=1)
x.columns


# In[ ]:


clf = RandomForestClassifier()
clf.fit(x, y)


# > <font size=4, color=blue>Accuracy</font>

# In[ ]:


clf.score(x,y)


# In[ ]:


clf.feature_importances_


# > <font size=4, color=blue>Feature Importance plot</font>

# In[ ]:


importances = clf.feature_importances_
indices = np.argsort(importances)
features=x.columns
fig, ax = plt.subplots(figsize=(9,9))
plt.title("Feature Impoprtance")
plt.ylabel("Features")
plt.barh(range(len(indices)), importances[indices] )
plt.yticks(range(len(indices)), [features[i] for i in indices])
print()


# ### Train test split 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# ### Random forest

# In[ ]:


clf.fit(X_train,y_train)


# > <font size=4, color=blue>Accuracy</font>

# In[ ]:


print("Train accuracy: ",clf.score(X_train,y_train))

print("Test accuracy: ",clf.score(X_test,y_test))


# ### Decision tree 

# In[ ]:


from sklearn import tree

dt=tree.DecisionTreeClassifier()
dt

dt.fit(X_train,y_train)
print("Train data accuracy:",dt.score(X_train,y_train))

print("Test data accuracy:",dt.score(X_test,y_test))


# ### One-Hot Encoding 

# In[ ]:


data=pd.read_csv("../../../input/becksddf_churn-in-telecoms-dataset/bigml_59c28831336c6604c800002a.csv")


# In[ ]:


data.head()


# In[ ]:


data.drop(["phone number"],axis=1,inplace =True)


# In[ ]:


data.dtypes


# In[ ]:


data.dtypes.value_counts()


# In[ ]:


cate


# > <font size=4, color=blue>get_dummies function</font>

# In[ ]:


enc=pd.get_dummies(data[cate[:-1]])


# In[ ]:


enc.columns


# In[ ]:


data.columns


# In[ ]:


data.drop(cate[:-1],axis=1,inplace=True)


# In[ ]:


data[enc.columns]=enc


# In[ ]:


data.shape


# In[ ]:


X=data.drop(["churn"],axis=1)
y=data["churn"]


# In[ ]:


X.shape


# ### Logistic Regression

# In[ ]:


lr=LogisticRegression().fit(X, y)
lr.score(X, y)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[ ]:


lr=LogisticRegression().fit(X_train, y_train)
print("Train accuracy:",lr.score(X_train, y_train))


# In[ ]:


print("Test accuracy:",lr.score(X_test,y_test))


# ### Saving to pkl file

# In[ ]:


X_train.columns


# In[ ]:


joblib.dump(X_train,'X_train.pkl') 
joblib.dump(y_train,'y_train.pkl') 
joblib.dump(X_test,'X_test.pkl') 
joblib.dump(y_test,'y_test.pkl')
joblib.dump(x,'x.pkl')
joblib.dump(y,'y.pkl')


# ### Load Pkl files

# In[ ]:


X_train=joblib.load('X_train.pkl')
y_train=joblib.load('y_train.pkl')
X_test=joblib.load('X_test.pkl')
y_test=joblib.load('y_test.pkl')
x=joblib.load('x.pkl')
y=joblib.load('y.pkl')


# ### All Classification Algorithm

# In[ ]:


algo = pd.DataFrame(columns=["Algorithm","Accuracy","auc score"])
algo.head()


# > ### LogisticRegression

# In[ ]:


clf = LogisticRegression(C=1.0)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("accuracy_score",accuracy_score(y_test, predictions))
print("auc",roc_auc_score(y_test, predictions))
lr = pd.Series([clf.__class__,accuracy_score(y_test, predictions),roc_auc_score(y_test, predictions)], ["Algorithm","Accuracy","auc score"]) 
algo=algo.append([lr],ignore_index=True)


# > ### XGBClassifier

# In[ ]:


clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, subsample=0.8, nthread=10, learning_rate=0.1) 
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("accuracy_score",accuracy_score(y_test, predictions))
print("auc",roc_auc_score(y_test, predictions))
lr = pd.Series([clf.__class__,accuracy_score(y_test, predictions),roc_auc_score(y_test, predictions)], ["Algorithm","Accuracy","auc score"]) 
algo=algo.append([lr],ignore_index=True)


# > ### MultinomialNB

# In[ ]:


clf = MultinomialNB()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("accuracy_score",accuracy_score(y_test, predictions))
print("auc",roc_auc_score(y_test, predictions))
lr = pd.Series([clf.__class__,accuracy_score(y_test, predictions),roc_auc_score(y_test, predictions)], ["Algorithm","Accuracy","auc score"]) 
algo=algo.append([lr],ignore_index=True)


# > ###  AdaBoostClassifier 

# In[ ]:


clf = AdaBoostClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("accuracy_score",accuracy_score(y_test, predictions))
print("auc",roc_auc_score(y_test, predictions))
lr = pd.Series([clf.__class__,accuracy_score(y_test, predictions),roc_auc_score(y_test, predictions)], ["Algorithm","Accuracy","auc score"]) 
algo=algo.append([lr],ignore_index=True)


# > ### KNeighborsClassifier

# In[ ]:


clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("accuracy_score",accuracy_score(y_test, predictions))
print("auc",roc_auc_score(y_test, predictions))
lr = pd.Series([clf.__class__,accuracy_score(y_test, predictions),roc_auc_score(y_test, predictions)], ["Algorithm","Accuracy","auc score"]) 
algo=algo.append([lr],ignore_index=True)


# > ### GradientBoostingClassifier

# In[ ]:


clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("accuracy_score",accuracy_score(y_test, predictions))
print("auc",roc_auc_score(y_test, predictions))
lr = pd.Series([clf.__class__,accuracy_score(y_test, predictions),roc_auc_score(y_test, predictions)], ["Algorithm","Accuracy","auc score"]) 
algo=algo.append([lr],ignore_index=True)


# > ### ExtraTreesClassifier

# In[ ]:


clf = ExtraTreesClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("accuracy_score",accuracy_score(y_test, predictions))
print("auc",roc_auc_score(y_test, predictions))
lr = pd.Series([clf.__class__,accuracy_score(y_test, predictions),roc_auc_score(y_test, predictions)], ["Algorithm","Accuracy","auc score"]) 
algo=algo.append([lr],ignore_index=True)


# > ### DecisionTreeClassifier

# In[ ]:


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("accuracy_score",accuracy_score(y_test, predictions))
print("auc",roc_auc_score(y_test, predictions))
lr = pd.Series([clf.__class__,accuracy_score(y_test, predictions),roc_auc_score(y_test, predictions)], ["Algorithm","Accuracy","auc score"]) 
algo=algo.append([lr],ignore_index=True)


# In[ ]:


algo.sort_values(["Accuracy"], ascending=[False])

