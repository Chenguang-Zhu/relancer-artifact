#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../../../input/primaryobjects_voicegender/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../../../input/primaryobjects_voicegender"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../../../input/primaryobjects_voicegender/voice.csv")


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# In[ ]:


df.shape


# In[ ]:


#For normally distributed data, the skewness should be about 0. 
#A skewness value > 0 means that there is more weight in the left tail of the distribution. 

df.skew()


# In[ ]:


#Finding Correlation among the features
df.corr()


# In[ ]:


df.isnull().sum()
#This shows that our data has no missing values in it. That is good!


# In[ ]:


print("Total number ber of people involve in the test: {}".format(df.shape[0]))
print("Number of Male: {}".format(df[df.label == 'male'].shape[0]))
print("Number of Female: {}".format(df[df.label == 'female'].shape[0]))

#It proves the data contain same number of male and female labels


# In[ ]:


#Visualising individual features of our data
df.hist(figsize=(16,16))
print()


# In[ ]:


#Finding the relationship between independent and dependent variable(Label)
sns.pointplot(x='label',y='meanfreq',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='sd',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='median',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='Q25',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='Q75',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='IQR',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='skew',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='kurt',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='sp.ent',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='sfm',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='mode',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='centroid',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='meanfun',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='minfun',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='maxfun',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='meandom',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='mindom',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='maxdom',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='dfrange',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


sns.pointplot(x='label',y='modindx',data=df,color='red',alpha=0.8,label = 'a')


# In[ ]:


X = df.iloc[:,:-1]
X.head()
#Now X contains all the features except labels


# In[ ]:


#Saving labels in variable y
y = df.iloc[:,-1]
y.head()


# In[ ]:


#Now we are encoding the labels in the form of ones and zeros
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)


# In[ ]:


print(y)
#It shows that our labels are encoded as ones and zeros
#One = Male
#Zero = Female


# In[ ]:


#Splitting our data into train and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[ ]:


#Applying feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[ ]:


#Now we are applying SVM algorithm and using Linear Kernal
from sklearn.svm import SVC
classifier = SVC(kernel='linear',random_state=0)
classifier.fit(X_train,y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


accuracy1 = ((286+328)/(286+328+12+8)) * 100
print(accuracy1)
#We got an accuracy of 96.37% by applying by applying Linear Kernel


# In[ ]:


#Now we are applying SVM algorithm and using RBF Kernal
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


accuracy2 = ((290+330)/(290+330+10+4)) * 100
print(accuracy2)
#We got an accuracy of 96.37% by applying by applying RBF Kernel


# In[ ]:


#Now we are applying SVM algorithm and using Polynomial Kernal
from sklearn.svm import SVC
classifier = SVC(kernel='poly',random_state=0)
classifier.fit(X_train,y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


accuracy3 = ((278+333)/(278+333+7+16)) * 100
print(accuracy3)
#We got an accuracy of 96.37% by applying by applying Polynomial Kernel


# In[ ]:


#Comparring results of different Kernals
objects = ('SVM-LINEAR', 'SVM-RBF', 'SVM-POLYNOMIAL')
y_pos = np.arange(len(objects))
performance = [accuracy1,accuracy2,accuracy3]
 
plt.scatter(y_pos, performance, alpha=1)
plt.plot(y_pos, performance,color='blue')
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy %')
plt.xticks(rotation=45)
plt.title('SVM Kernals Accuracy')
print()


# In[ ]:


#In conclusion
print("We got an accuracy of {}".format(accuracy1),"by applying SVM using Linear Kernal ")
print("We got an accuracy of {}".format(accuracy2),"by applying SVM using RBF Kernal ")
print("We got an accuracy of {}".format(accuracy3),"by applying SVM using Polynomial Kernal ")


