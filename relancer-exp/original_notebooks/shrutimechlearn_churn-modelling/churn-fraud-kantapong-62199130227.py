#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../../../input/shrutimechlearn_churn-modelling/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk("../../../input/shrutimechlearn_churn-modelling"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print()


# In[ ]:


df = pd.read_csv("../../../input/shrutimechlearn_churn-modelling/Churn_Modelling.csv")


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


X=df.loc[:,['CreditScore', 'Age', 'Tenure', 'Balance' , 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']].values
y=df.iloc[:,-1].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.25,random_state =0)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X =StandardScaler()
X_train =sc_X.fit_transform(X_train)
X_test =sc_X.transform(X_test)


# svm

# In[ ]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'poly', degree = 2, random_state = 0) #degree for non-linear
classifier.fit(X_train, y_train) 


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# K-nearest neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# <h2>Machine Learning</h2>

# In[ ]:


X = df.iloc[:, 3:13].values
y = df.iloc[:, 13].values


# In[ ]:


print(X.shape)


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

transformer = ColumnTransformer( transformers=[ ("OneHot", OneHotEncoder(), [1,2] ) ] ) 

X2 = transformer.fit_transform(X)
print(X2.shape)
print(X2[0:20, :])


# In[ ]:


print(X2.shape)
print(X2[0:10, :])


# In[ ]:


X = np.concatenate((X[:,0:1],X[:,3:10], X2[:,1:4]), axis=1)
print(X.shape)
print(X[0:5, :])


# <h2>Start Building Neural Network</h2>

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


# Initialising the ANN
classifier = Sequential()


# In[ ]:


# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))


# In[ ]:


# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))


# In[ ]:


# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


# In[ ]:


# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


classifier.summary()


# In[ ]:




