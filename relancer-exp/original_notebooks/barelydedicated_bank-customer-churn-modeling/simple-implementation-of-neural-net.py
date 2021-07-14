#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../../../input/barelydedicated_bank-customer-churn-modeling/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk("../../../input/barelydedicated_bank-customer-churn-modeling"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


bank_data = pd.read_csv("../../../input/barelydedicated_bank-customer-churn-modeling/Churn_Modelling.csv")


# In[ ]:


bank_data.describe()


# In[ ]:


bank_data.head()


# In[ ]:


bank_data.info()


# In[ ]:


bank_data_new = bank_data.drop(['RowNumber', 'CustomerId', 'Surname'], axis =1)


# In[ ]:


numerical_distribution = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
for i in numerical_distribution:
    plt.hist(bank_data_new[i])
    plt.title(i)
    print()


# In[ ]:


bank_data_new.head()


# In[ ]:


#splitting data into features and target variable
X = bank_data_new.iloc[:,:-1].values
y = bank_data_new.iloc[:,-1].values


# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X_1 = LabelEncoder()
X[:, 1]= labelEncoder_X_1.fit_transform(X[:, 1])

labelEncoder_X_2 = LabelEncoder()
X[:, 2] = labelEncoder_X_2.fit_transform(X[:, 2])


# In[ ]:


X


# In[ ]:


oneHotEncoder = OneHotEncoder(categorical_features = [1])
X = oneHotEncoder.fit_transform(X).toarray()


# In[ ]:


X


# In[ ]:


X = X[:, 1:]


# In[ ]:


#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state= 42)


# In[ ]:


#feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


#Using sequential model using Keras
import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


#building ANN classifier

classifier = Sequential()

#adding first layer(input layer)
classifier.add(Dense(units=6, kernel_initializer='uniform', input_dim=11, activation='relu')) 

#adding second layer(hidden layer)
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

#adding third layer(output layer)
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

#compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#train the ANN
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


y_pred = (y_pred > 0.5)


# In[ ]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)


# In[ ]:


cm

