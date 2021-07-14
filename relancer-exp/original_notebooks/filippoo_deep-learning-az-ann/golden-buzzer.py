#!/usr/bin/env python
# coding: utf-8

# ## Churn_Modelling_Golden_Prediction

# In this kernel I am going to make an Exploratory Data Analysis (EDA) on this dataset. Also I am going to make different predictive models and find out the best one with highest prediction accuracy.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../../../input/filippoo_deep-learning-az-ann/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk("../../../input/filippoo_deep-learning-az-ann"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Importing the Churn dataset

# In[ ]:


df = pd.read_csv("../../../input/filippoo_deep-learning-az-ann/Churn_Modelling.csv")


# # DataUnderstanding and Pre-processig

# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.info


# In[ ]:


df.describe()


# In[ ]:


df.columns


# # Spitting the data into independent and dependent variables

# In[ ]:


# Independent variables
X = df.iloc[:,3:13]
# Dependent variable
y = df.iloc[:,13]


# In[ ]:


y


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


X.head()


# In[ ]:


df.columns


# In[ ]:


dummy=pd.get_dummies(X[[ 'Geography','Gender']])
X=pd.concat([X,dummy],axis=1)




# In[ ]:


X.drop([ 'Geography','Gender'],axis = 1,inplace = True)


# In[ ]:


X.head()


# In[ ]:


print()
# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.3,random_state=1)


# In[ ]:


print("X_train :-{} \nX_test :-{}\ny_train :-{}\ny_test :-{}".format(X_train.shape, X_test.shape,y_train.shape,y_test.shape)) 


# In[ ]:


X_train.astype(int)


# In[ ]:





# In[ ]:


y_train


# In[ ]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # Importing the keras libraries and packages

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# **Initialising the ANN **

# In[ ]:


classifier = Sequential()


# Adding the input layer and first hidden layers in ANN

# In[ ]:


classifier.add(Dense(output_dim=6,init ='uniform',activation = 'relu',input_dim =13))
classifier.add(Dropout(p=0.1))


# Second hidden layers in ANN structure

# In[ ]:


classifier.add(Dense(output_dim=6,init ='uniform',activation = 'relu'))
classifier.add(Dropout(p=0.1))


# Adding output layers

# In[ ]:


classifier.add(Dense(output_dim=1,init ='uniform',activation = 'sigmoid'))


# Compiling the ANN model

# In[ ]:


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# Fit the model

# In[ ]:


classifier.fit(X_train,y_train,epochs=100,batch_size=10)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


y_pred = (y_pred > 0.5)


# In[ ]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# Evaulating the ANN model

# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


# In[ ]:


def buildclassifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=6,init ='uniform',activation = 'relu',input_dim =13))
    classifier.add(Dense(output_dim=6,init ='uniform',activation = 'relu'))
    classifier.add(Dense(output_dim=1,init ='uniform',activation = 'sigmoid'))
    classifier.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier


# In[ ]:


classifier = KerasClassifier(build_fn=buildclassifier,epochs=500,batch_size=10)


# In[ ]:


accuracies = cross_val_score(estimator=classifier,X = X_train,y =y_train,cv=10,n_jobs=-1)


# In[ ]:


accuracies


# In[ ]:


mean = accuracies.mean()
mean


# In[ ]:


std = accuracies.std()
std


# In[ ]:


''' from keras.wrappers.scikit_learn import KerasClassifier from sklearn.model_selection import GridSearchCV def buildclassifier(optimize): classifier = Sequential() classifier.add(Dense(output_dim=6,init ='uniform',activation = 'relu',input_dim =13)) classifier.add(Dense(output_dim=6,init ='uniform',activation = 'relu')) classifier.add(Dense(output_dim=1,init ='uniform',activation = 'sigmoid')) classifier.compile(optimizer=optimize,loss='binary_crossentropy',metrics=['accuracy']) return classifier classifier = KerasClassifier(build_fn=buildclassifier) parameters = { 'batch_size':[25,34], 'epochs':[100,500], 'optimize': ['adam','rmsprop']} gridsearch = GridSearchCV(estimator=classifier, param_grid=parameters, cv =10, scoring='accuracy')  gridsearch = gridsearch.fit(X_train,y_train)  best_param = gridsearch.best_params_ best_accuracy = gridsearch.best_score_    ''' 


# In[ ]:




