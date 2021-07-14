#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../../../input/ronitf_heart-disease-uci/" directory.z
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../../../input/ronitf_heart-disease-uci/"))

# Any results you write to the current directory are saved as output.


# > # If you find this notebook helpful , some upvotes would be very much appreciated - That will keep me motivated 👍

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from eli5.sklearn import PermutationImportance
import warnings
#perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
#eli5.show_weights(perm, feature_names = X_test.columns.tolist())
warnings.filterwarnings("ignore")
sns.set_style("darkgrid")


# In[ ]:


df=pd.read_csv("../../../input/ronitf_heart-disease-uci/heart.csv")


# In[ ]:


df.head()


# * age- in years
# * sex-(1 = male; 0 = female)
# * cp- chest pain type
# * trestbps- resting blood pressure (in mm Hg on admission to the hospital)
# * chol- serum cholestoral in mg/dl
# * fbs-(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# * restecg-resting electrocardiographic results
# * thalach-maximum heart rate achieved
# * exang-exercise induced angina (1 = yes; 0 = no)
# * oldpeak-ST depression induced by exercise relative to rest
# * slope-the slope of the peak exercise ST segment
# * ca-number of major vessels (0-3) colored by flourosopy
# * thal- 3 = normal; 6 = fixed defect; 7 = reversable defect
# * target- 1 or 0

# In[ ]:


df.info()


# In[ ]:


plt.figure(figsize=(10,8))
#No much of correlation


# In[ ]:


df['target'].value_counts()


# In[ ]:


sns.distplot(df['age'],color='Red',hist_kws={'alpha':1,"linewidth": 2}, kde_kws={"color": "k", "lw": 3, "label": "KDE"})
#Most people age is from 40 to 60


# In[ ]:


fig,ax=plt.subplots(figsize=(24,6))
plt.subplot(1, 3, 1)
age_bins = [20,30,40,50,60,70,80]
df['bin_age']=pd.cut(df['age'], bins=age_bins)
g1=sns.countplot(x='bin_age',data=df ,hue='target',palette='plasma',linewidth=3)
g1.set_title("Age vs Heart Disease")
#The number of people with heart disease are more from the age 41-55
#Also most of the people fear heart disease and go for a checkup from age 55-65 and dont have heart disease (Precautions)

plt.subplot(1, 3, 2)
cho_bins = [100,150,200,250,300,350,400,450]
df['bin_chol']=pd.cut(df['chol'], bins=cho_bins)
g2=sns.countplot(x='bin_chol',data=df,hue='target',palette='plasma',linewidth=3)
g2.set_title("Cholestoral vs Heart Disease")
#Most people get the heart disease with 200-250 cholestrol 
#The others with cholestrol of above 250 tend to think they have heart disease but the rate of heart disease falls

plt.subplot(1, 3, 3)
thal_bins = [60,80,100,120,140,160,180,200,220]
df['bin_thal']=pd.cut(df['thalach'], bins=thal_bins)
g3=sns.countplot(x='bin_thal',data=df,hue='target',palette='plasma',linewidth=3)
g3.set_title("Thal vs Heart Disease")
#People who have thalach between 140-180 have a very high chance of getting the heart disease 


# In[ ]:


fig,ax=plt.subplots(figsize=(24,6))
plt.subplot(131)
x1=sns.countplot(x='cp',data=df,hue='target',palette='spring',linewidth=3)
x1.set_title('Chest pain type')
#Chest pain type 2 people have highest chance of heart disease

plt.subplot(132)
x2=sns.countplot(x='thal',data=df,hue='target',palette='spring',linewidth=3)
x2.set_title('Thal')
#People with thal 2 have the highest chance of heart disease

plt.subplot(133)
x3=sns.countplot(x='slope',data=df,hue='target',palette='spring',linewidth=3)
x3.set_title('slope of the peak exercise ST segment')
#Slope 2 people have higher chance of heart disease


# In[ ]:


fig,ax=plt.subplots(figsize=(16,6))
plt.subplot(121)
s1=sns.boxenplot(x='sex',y='age',hue='target',data=df,palette='YlGn',linewidth=3)
s1.set_title("Figure 1")
#Figure 1 says most of females having heart disease range from 40-70yrs and men from 40-60yrs

plt.subplot(122)
s2=sns.pointplot(x='sex',y='age',hue='target',data=df,palette='autumn',capsize=.2)
s2.set_title("Figure 2")
#Figure 2 says mean age for female with heart disease around 54yrs and for males around 51yrs


# In[ ]:


fig,ax=plt.subplots(figsize=(16,6))
sns.pointplot(x='age',y='cp',data=df,color='Lime',hue='target',linestyles=["-", "--"])
plt.title('Age vs Cp')
#People with heart disease tend to have higher 'cp' at all ages only exceptions at age 45 and 49


# In[ ]:


fig,ax=plt.subplots(figsize=(16,6))
sns.lineplot(y='thalach',x='age',data=df,hue="target",style='target',palette='magma',markers=True, dashes=False,err_style="bars", ci=68)
plt.title('Age vs Thalach')
#Thalach always high in people having heart disease and as age increases the thalach seems to reduce and other factors might play a role in heart disease


# In[ ]:


sns.pointplot(x='sex',y='thal',data=df,hue='target',markers=["o", "x"],linestyles=["-", "--"],capsize=.2,palette='coolwarm')
#Both males and females without heart disease have higher thal value and males with heart diseases tend to have higher thal than females


# In[ ]:


sns.countplot(x='ca',data=df,hue='target',palette='YlOrRd',linewidth=3)
# People with 'ca' as 0 have highest chance of heart disease


# In[ ]:


sns.countplot(x='slope',hue='target',data=df,palette='bwr',linewidth=3)
#Slope 2 has highest people with heart disease


# In[ ]:


fig,ax=plt.subplots(figsize=(24,6))
plt.subplot(131)
old_bins = [0,1,2,3,4,5,6]
df['bin_old']=pd.cut(df['oldpeak'], bins=old_bins)
sns.countplot(x='bin_old',hue='target',data=df,palette='hot',linewidth=3)
plt.title("Figure 1")
#Figure 1: As the value of oldpeak increases the rate of heart disease decreases

plt.subplot(132)
sns.boxplot(x='slope',y='oldpeak',data=df,hue='target',palette='hot',linewidth=3)
plt.title("Figure 2")
#Figure 2: slope-s and target = 1; for s=0 --> Median Oldpeak=~1.4; for s=1 --> Median Oldpeak=~0.7; for s=2 --> Median Oldpeak=~0

plt.subplot(133)
sns.pointplot(x='slope',y='oldpeak',data=df,hue='target',palette='hot')
plt.title("Figure 3")
#Figure 3: As the value of slope increases the oldpeak values decrease and heart disease people have lower oldpeak


# In[ ]:


df.head()


# In[ ]:


df.drop(['bin_age','bin_chol','bin_thal','bin_old'],axis=1,inplace=True)


# In[ ]:


df.head()


# # Modelling 
# 

# ## Data Cleaning

# In[ ]:


#Outlier Detection

from collections import Counter
def detect_outliers(df,n,features):
    """    Takes a dataframe df of features and returns a list of the indices    corresponding to the observations containing more than n outliers according    to the Tukey method.    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(df,2,['trestbps', 'chol','thalach'])


# In[ ]:


df.loc[Outliers_to_drop] # Show the outliers rows

#No outliers to drop as the values of all the columns are in ranges.


# In[ ]:


#Checking Missing Data
df.isnull().sum()
#No missing data


# In[ ]:


df.head()


# In[ ]:


#df=pd.read_csv("../../../input/ronitf_heart-disease-uci/heart.csv")


# In[ ]:


df.dtypes


# In[ ]:


#Conversion to categorical variables
df['sex']=df['sex'].astype('category')
df['cp']=df['cp'].astype('category')
df['fbs']=df['fbs'].astype('category')
df['restecg']=df['restecg'].astype('category')
df['exang']=df['exang'].astype('category')
df['slope']=df['slope'].astype('category')
df['ca']=df['ca'].astype('category')
df['thal']=df['thal'].astype('category')
df['target']=df['target'].astype('category')
df.dtypes


# In[ ]:


y=df['target']


# In[ ]:


df=pd.get_dummies(df,drop_first=True)
df.head()


# In[ ]:


X=df.drop('target_1',axis=1)
X.head()


# In[ ]:


X.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
#from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

classifiers=[['Logistic Regression :',LogisticRegression()],['Decision Tree Classification :',DecisionTreeClassifier()],['Random Forest Classification :',RandomForestClassifier()],['Gradient Boosting Classification :', GradientBoostingClassifier()],['Ada Boosting Classification :',AdaBoostClassifier()],['Extra Tree Classification :', ExtraTreesClassifier()],['K-Neighbors Classification :',KNeighborsClassifier()],['Support Vector Classification :',SVC()],['Gaussian Naive Bayes :',GaussianNB()]]
cla_pred=[]
for name,model in classifiers:
    model=model
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    cla_pred.append(accuracy_score(y_test,predictions))
    print(name,accuracy_score(y_test,predictions))


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix

logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
log_pred=logmodel.predict(X_test)
print(confusion_matrix(y_test,log_pred))
print(classification_report(y_test,log_pred))
print(accuracy_score(y_test,log_pred))


# In[ ]:


#Hyperparameter tuning for Logistic Regression
from sklearn.model_selection import GridSearchCV
penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
hyperparameters = dict(C=C, penalty=penalty)
h_logmodel = GridSearchCV(logmodel, hyperparameters, cv=5, verbose=0)
best_logmodel=h_logmodel.fit(X,y)
print('Best Penalty:', best_logmodel.best_estimator_.get_params()['penalty'])
print('Best C:', best_logmodel.best_estimator_.get_params()['C'])


# In[ ]:


logmodel=LogisticRegression(penalty='l1',C=2.7825594022071245)
logmodel.fit(X_train,y_train)
h_log_pred=logmodel.predict(X_test)
print(confusion_matrix(y_test,h_log_pred))
print(classification_report(y_test,h_log_pred))
print(accuracy_score(y_test,h_log_pred))

#3% increase in the accuracy!!


# # So on Hyperparameter tuning we get a model with  Logistic Regression with 90% accuray!!

# ### Always open to suggestions and Thank you for you're time.

