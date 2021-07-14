#!/usr/bin/env python
# coding: utf-8

# **Table of content:-**
# 
# 1.Importing Dataset
# 
# 2.Dataset Visualization
# 
# 3.Dataset Modeling
# 
#   1. Countplot
# 
#   2. Boxplot
# 
#   3. Pairplot
# 
#   4. Histogram
# 
#   5. Violinplot
# 
#   6. Distplot
# 
#   7. KDE plot
#   
#   8. Heatmap

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# **Problem statement:-**
# 
# The dataset contains factor affecting placements in the college.
# 
# **Data columns**:-
# 
# 1)sl_no	:- Serial Number
# 
# 2)gender:- Male or Female
# 
# 3)ssc_p	:- SSC percentage
# 
# 4)ssc_b	:- SSC board
# 
# 4)hsc_p	:- HSC percentage
# 
# 5)hsc_b	:- HSC board
# 
# 6)hsc_s :- HSC specialization
# 
# 7)degree_p	:- Degree percentage
# 
# 8)degree_t	:- Degree specialization
# 
# 9)workex	:- work experience
# 
# 10)etest_p :- Etest percentage
# 
# 11)specialisation	
# 
# 12)mba_p:-MBA percentage
# 
# 13)status:- placed or not placed
# 
# 14)salary
# 

# In[ ]:


df=pd.read_csv("../../../input/benroshan_factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
df.head()


# In[ ]:


df.dtypes


# In[ ]:


df.isnull().sum()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.info()


# **Heatmap**

# In[ ]:


plt.figure(figsize=(12, 6))
print()


# **Box plot**

# In[ ]:


f,ax=plt.subplots(2,2,figsize=(25,20))
sns.boxplot(x="salary",ax=ax[0][0],data=df, palette="muted")
sns.boxplot(x="degree_p",data=df,ax=ax[0][1], palette="muted")
sns.boxplot(x="hsc_p",ax=ax[1][0],data=df, palette="muted")
sns.boxplot(x="ssc_p",ax=ax[1][1],data=df, palette="muted")


# **Countplot**

# In[ ]:


f,ax=plt.subplots(3,2,figsize=(25,25))
sns.countplot(x="specialisation",ax=ax[0][0],data=df,hue="status",palette="muted")
sns.countplot(x="status",data=df,ax=ax[0][1], palette="muted")
sns.countplot(x="degree_t",ax=ax[1][0],data=df,hue="status", palette="muted")
sns.countplot(x="hsc_s",ax=ax[1][1],data=df,hue="status", palette="muted")
sns.countplot(x="ssc_b",ax=ax[2][0],data=df,hue="status", palette="muted")
sns.countplot(x="gender",ax=ax[2][1],data=df, hue="status",palette="muted")


# **Distplot**

# In[ ]:


ax = sns.distplot(df['etest_p'], rug=True, hist=True)


# **Violinplot**

# In[ ]:


ax = sns.violinplot(x="etest_p", y="status", data=df, palette="muted")


# **KDE Plot**

# In[ ]:


f,ax=plt.subplots(1,1,figsize=(25,6))
sns.kdeplot(df.loc[(df['status']=='Placed'), 'etest_p'], color='r', shade=True, Label='Placed')
sns.kdeplot(df.loc[(df['status']=='Not Placed'), 'etest_p'], color='g', shade=True, Label='Not Placed')
plt.xlabel('Etest_p') 


# **Histogram**

# In[ ]:


df1=df.drop(['sl_no'], axis=1)
df1.hist (bins=10,figsize=(20,20))
print ()


# **Pairplot**

# In[ ]:


sns.set(style="ticks", color_codes=True)
print()


# In[ ]:


print()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix


# In[ ]:


df.loc[df['status']=='Not Placed', 'status'] = 0
df.loc[df['status']=='Placed', 'status'] = 1
df.loc[df['degree_t']=='Sci&Tech','degree_t'] = 0
df.loc[df['degree_t']=='Comm&Mgmt','degree_t'] = 1
df.loc[df['degree_t']=='Others','degree_t'] = 2
df.loc[df['gender']=='M','gender'] = 0
df.loc[df['gender']=='F','gender'] = 1
df.loc[df['ssc_b']=='Others','ssc_b'] = 0
df.loc[df['ssc_b']=='Central','ssc_b'] = 1
df.loc[df['hsc_b']=='Others','hsc_b'] = 0
df.loc[df['hsc_b']=='Central','hsc_b'] = 1
df.loc[df['ssc_b']=='Others','ssc_b'] = 0
df.loc[df['ssc_b']=='Central','ssc_b'] = 1
df.loc[df['workex']=='No','workex'] = 0
df.loc[df['workex']=='Yes','workex'] = 1

df['workex'].astype(int)
df['ssc_b'].astype(int)
df['hsc_b'].astype(int)
df['gender'].astype(int)
df['degree_t'].astype(int)



# **Traintest split**

# In[ ]:


x = df.drop(['status','salary','sl_no','hsc_s','specialisation'],axis=1)
y=df['status'].astype(int)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=3)


# In[ ]:


seed=7
models = []
models.append(('RF',RandomForestClassifier()))
models.append(('SVM',SVC()))
models.append(('LR',LogisticRegression()))
models.append(('NB',GaussianNB()))
# Evaluating each models in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=seed)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)


# **Logistic Regression**

# In[ ]:


logistic = LogisticRegression()
logistic.fit(x_train,y_train)
y_pred=logistic.predict(x_test)
print(classification_report(y_test,y_pred))
accuracy1=logistic.score(x_test,y_test)
print (accuracy1*100,'%')
cm = confusion_matrix(y_test, y_pred)
print()


# **SVM**

# In[ ]:


classifier=SVC()
classifier.fit(x_train,y_train)
svm_predict=classifier.predict(x_test)
print(classification_report(y_test,svm_predict))
accuracy2=classifier.score(x_test,y_test)
print(accuracy2*100,'%')
cm = confusion_matrix(y_test, svm_predict)
print()


# **Randon forest classifier**

# In[ ]:


ran_class=RandomForestClassifier()
ran_class.fit(x_train,y_train)
ran_predict=ran_class.predict(x_test)
print(classification_report(y_test,ran_predict))
accuracy3=ran_class.score(x_test,y_test)
print(accuracy3*100,'%')
cm = confusion_matrix(y_test, ran_predict)
print()


# In[ ]:


# Defining the decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(x,y)

print('Decision Tree Classifer Created')


# In[ ]:


# Install required libraries


# In[ ]:


feature_names=x.columns


# In[ ]:


# Import necessary libraries for graph viz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

# Visualize the graph
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, feature_names=x.columns, filled=True, rounded=True, special_characters=True) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

