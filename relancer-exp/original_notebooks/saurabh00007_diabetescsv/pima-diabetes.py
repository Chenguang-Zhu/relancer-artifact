#!/usr/bin/env python
# coding: utf-8

# Pima Diabetes Analysis

# In[ ]:


#Loading all libs
# Importing the libraries
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print()
plt.style.use('fivethirtyeight')


# In[ ]:


df=pd.read_csv("../../../input/saurabh00007_diabetescsv/diabetes.csv")


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


#Finding and visualizing the corelation of features
df_cr=df.corr()
plt.figure(figsize=(10,6))
print()
print()


# In[ ]:


df_n2=df[['Pregnancies','Glucose','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']]
#df_n2=pd.DataFrame(df.iloc[:,0:8])


# In[ ]:


#Finding and visualizing the corelation of featuresdf_cr=df.corr()
df_corelation=df_n2.corr()


# In[ ]:


plt.figure(figsize=(10,6))
print()


# In[ ]:


#Elimanting Outliers using ZScore
from scipy import stats
df_new=df_n2[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
# Comparing the two dataframes
df_new.describe()


# In[ ]:


df_n2.describe()


# In[ ]:


x1=df_new.iloc[:,0:6].values  #Independent features
y1=df_new.iloc[:,-1].values  #Output variables
#x1=df.iloc[:,0:6].values  #Independent features
#y1=df.iloc[:,-1].values  #Output variables
x1.shape,y1.shape
plt.boxplot(x1,vert=False,labels=['Pregnancies', 'Glucose','Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'],patch_artist=True)
print()


# In[ ]:


# Importing libraries
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
np.random.seed(101)


# In[ ]:


#Splitting the dataset in train and test
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.30)


# In[ ]:


mms=MinMaxScaler()
sc=StandardScaler()


# In[ ]:


# Standardizing the data
x_train_mms=mms.fit_transform(x_train)
x_test_mms=mms.fit_transform(x_test)
x_train_sc=sc.fit_transform(x_train)
x_test_sc=sc.fit_transform(x_test)
plt.boxplot(x_train_mms,vert=False,labels=['Pregnancies', 'Glucose', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'], patch_artist=True) 
print()
plt.boxplot(x_test_mms,vert=False,labels=['Pregnancies', 'Glucose', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'], patch_artist=True) 
print()
plt.boxplot(x_train_sc,vert=False,labels=['Pregnancies', 'Glucose', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'], patch_artist=True) 
print()
plt.boxplot(x_test_sc,vert=False,labels=['Pregnancies', 'Glucose', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'], patch_artist=True) 
print()


# In[ ]:


model_logistic_mms=LogisticRegression(C=10)
model_knn_mms=KNeighborsClassifier(n_neighbors=5)
model_svc_mms= SVC(C=10,kernel='rbf',probability=True)
model_dt_mms=DecisionTreeClassifier()
model_rf_mms=RandomForestClassifier(n_estimators=100)
model_logistic_sc=LogisticRegression(C=10)
model_knn_sc=KNeighborsClassifier(n_neighbors=5)
model_svc_sc= SVC(C=10,kernel='rbf',probability=True)
model_dt_sc=DecisionTreeClassifier()
model_rf_sc=RandomForestClassifier(n_estimators=100)
model_pca=PCA()


# In[ ]:


model_logistic_mms.fit(x_train_mms,y_train)
model_knn_mms.fit(x_train_mms,y_train)
model_svc_mms.fit(x_train_mms,y_train)
model_dt_mms.fit(x_train_mms,y_train)
model_rf_mms.fit(x_train_mms,y_train)


# In[ ]:


model_logistic_sc.fit(x_train_sc,y_train)
model_knn_sc.fit(x_train_sc,y_train)
model_svc_sc.fit(x_train_sc,y_train)
model_dt_sc.fit(x_train_sc,y_train)
model_rf_sc.fit(x_train_sc,y_train)
model_pca.fit(x_train_sc)


# In[ ]:


variance=model_pca.explained_variance_


# In[ ]:


prob_score=model_pca.explained_variance_ratio_


# In[ ]:


x_pca=model_pca.fit_transform(x_train_sc)


# In[ ]:


x_pca.shape


# In[ ]:


plt.bar(range(0,6),prob_score)


# In[ ]:


prob_score


# In[ ]:


#Cumulutative plot
plt.plot(np.cumsum(prob_score))    
print()


# In[ ]:


pca_1=PCA(0.5)
pca_1.fit(x_train_sc)
num_com=pca_1.n_components_
num_com


# In[ ]:


x_train_pca=pca_1.fit_transform(x_train_sc)
x_test_pca=pca_1.fit_transform(x_test_sc)


# In[ ]:


model_pca_log=LogisticRegression(C=10)
model_pca_log.fit(x_train_pca,y_train)


# In[ ]:


y_pred_log=model_logistic_mms.predict(x_test_mms)
y_pred_knn=model_knn_mms.predict(x_test_mms)
y_pred_svc=model_svc_mms.predict(x_test_mms)
y_pred_dt=model_dt_mms.predict(x_test_mms)
y_pred_rf=model_rf_mms.predict(x_test_mms)
y_pred_pca_log=model_pca_log.predict(x_test_pca)


# In[ ]:


y_pred_log_sc=model_logistic_sc.predict(x_test_sc)
y_pred_knn_sc=model_knn_sc.predict(x_test_sc)
y_pred_svc_sc=model_svc_sc.predict(x_test_sc)
y_pred_dt_sc=model_dt_sc.predict(x_test_sc)
y_pred_rf_sc=model_rf_sc.predict(x_test_sc)


# In[ ]:


cm_log=confusion_matrix(y_test,y_pred_log)
cr_log=classification_report(y_test,y_pred_log)
cm_knn=confusion_matrix(y_test,y_pred_knn)
cr_knn=classification_report(y_test,y_pred_knn)
cm_svc=confusion_matrix(y_test,y_pred_svc)
cr_svc=classification_report(y_test,y_pred_svc)
cm_dt=confusion_matrix(y_test,y_pred_dt)
cr_dt=classification_report(y_test,y_pred_dt)
cm_rf=confusion_matrix(y_test,y_pred_rf)
cr_rf=classification_report(y_test,y_pred_rf)
cm_pca=confusion_matrix(y_test,y_pred_pca_log)
cr_pca=classification_report(y_test,y_pred_pca_log)


# In[ ]:


cm_log_sc=confusion_matrix(y_test,y_pred_log_sc)
cr_log_sc=classification_report(y_test,y_pred_log_sc)
cm_knn_sc=confusion_matrix(y_test,y_pred_knn_sc)
cr_knn_sc=classification_report(y_test,y_pred_knn_sc)
cm_svc_sc=confusion_matrix(y_test,y_pred_svc_sc)
cr_svc_sc=classification_report(y_test,y_pred_svc_sc)
cm_dt_sc=confusion_matrix(y_test,y_pred_dt_sc)
cr_dt_sc=classification_report(y_test,y_pred_dt_sc)
cm_rf_sc=confusion_matrix(y_test,y_pred_rf_sc)
cr_rf_sc=classification_report(y_test,y_pred_rf_sc)


# In[ ]:


print('=====Log Reg========')
plt.figure(figsize=(8,4))
print()
print()
print('=====KNN========')
plt.figure(figsize=(8,4))
print()
print()
print('=====SVC========')
plt.figure(figsize=(8,4))
print()
print()
print('=====Discreet Tree Class========')
plt.figure(figsize=(8,4))
print()
print()
print('=====Random Forest========')
plt.figure(figsize=(8,4))
print()
print()
print('=====PCA LOG========')
plt.figure(figsize=(8,4))
print()
print()


# In[ ]:


print('=====cr_log========')
print(cr_log)
print('=====cr_knn=========')
print(cr_knn)
print('=====cr_svc========')
print(cr_svc)
print('=====cr_dt=========')
print(cr_dt)
print('=====cr_rf========')
print(cr_rf)
print('=====cr_pca_log========')
print(cr_pca)


# In[ ]:


print('=====cr_log========')
print(cr_log_sc)
print('=====cr_knn=========')
print(cr_knn_sc)
print('=====cr_svc========')
print(cr_svc_sc)
print('=====cr_dt=========')
print(cr_dt_sc)
print('=====cr_rf========')
print(cr_rf_sc)

