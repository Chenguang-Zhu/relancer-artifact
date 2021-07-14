#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../../../input/filippoo_deep-learning-az-ann/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk("../../../input/filippoo_deep-learning-az-ann"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


dataset=pd.read_csv("../../../input/filippoo_deep-learning-az-ann/Churn_Modelling.csv")
dataset.columns
dataset.shape


# In[ ]:


features=["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Exited"]
dataset=dataset[features]
dataset.head(10)


# In[ ]:


X=dataset.iloc[:,:-1]
X.head()


# In[ ]:


Y=dataset.iloc[:,10]
Y.head()


# In[ ]:


X.isnull().sum()
#No null values in X


# In[ ]:


Y.isnull().sum()
#No null values in Y


# In[ ]:


X.Gender.head()


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1,2])],remainder="passthrough")
X=np.array(ct.fit_transform(X))


# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
xtrain=scaler.fit_transform(xtrain)
xtest=scaler.fit_transform(xtest)


# In[ ]:


#implementing neural networks
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier=Sequential()
classifier.add(Dense(input_dim=13,output_dim=6,kernel_initializer="uniform",activation='relu'))
classifier.add(Dense(output_dim=6,kernel_initializer="uniform",activation='relu'))
classifier.add(Dense(output_dim=6,kernel_initializer="uniform",activation='relu'))
classifier.add(Dense(output_dim=1,kernel_initializer="uniform",activation="sigmoid"))
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
classifier.fit(xtrain,ytrain,batch_size=25,epochs=100)
ypred=classifier.predict(xtest)


# In[ ]:


ypred=(ypred>0.5)
from sklearn.metrics import confusion_matrix
mat=confusion_matrix(ytest,ypred)
ytest.shape



# In[ ]:


print(mat)


# In[ ]:


(1550+133)/2000*100


# In[ ]:


#Hence our final accuracy of neural network is 84.15

