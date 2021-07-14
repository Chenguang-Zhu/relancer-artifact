#!/usr/bin/env python
# coding: utf-8

# Created by: Sangwook Cheon
# 
# Date: Dec 23, 2018
# 
# This is step-by-step guide to Artificial Neural Networks (ANN), which I created for reference. I added some useful notes along the way to clarify things. This notebook's content is from A-Z Datascience course, and I hope this will be useful to those who want to review materials covered, or anyone who wants to learn the basics of ANN.
# 
# ## Content:
# ### IMPLEMENTATION
# ### 1. Data preprocessing
# ### 2. Build the Keras model
# ### 3. Compile and fit the model
# ### 4. Make predictions and determine accuracy

# # Some notes on ANNs
# 
# ## The Neuron
# Axon: Transmitters of signals
# Dentrites: Receivers of signals
# 
# The ANNs imitate the behavior of human brain. Each neuron receives certain inputs from the previous neurons, and process that information to send signals to others.

# # The Activation Function
# 
# Options:
# * Threshold Function
# * Sigmoid Function
# * Rectified Linear Unit (ReLU)
# * Tanh
# 
# For binary classification, Threhold Function or Sigmoid Function should be used.
# It is common to apply ReLU to hidden layers, and Sigmoid to the final layer to produce results.
# 
# # Dataset overview (used in this kernel)
# A bank is trying to see whether or not customers will be leaving the bank, based on various information about each customer. These features include Credit Score, Gender, Balance, etc. (Please see the view of the dataset below). We will apply ANN to find meaningful correlations between these independent variables, and determine if a customer will leave or stay in the bank.

# ### 1. Data processing
# Data processing is crucial for ANNs to work properly. All steps are required, including feature scaling.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("../../../input/barelydedicated_bank-customer-churn-modeling/Churn_Modelling.csv")

#include relevant columns within x and y
x = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]
dataset.head()


# In[ ]:


x.head()


# In[ ]:


#deal with categorical data --> encode them

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x.iloc[:, 1] = labelencoder_x.fit_transform(x.iloc[:, 1]) #applying on Geography
x.head()


# In[ ]:


#apply encoder on Gender as well
labelencoder_x_2 = LabelEncoder()
x.iloc[:, 2] = labelencoder_x_2.fit_transform(x.iloc[:, 2]) #applying on Gender
x.head()


# In[ ]:


#One hot encoding. 

from keras.utils import to_categorical
encoded = pd.DataFrame(to_categorical(x.iloc[:, 1]))
#no need to encode Gender, as there are only two categories

x = pd.concat([encoded, x], axis = 1)
x.head()


# In[ ]:


#Dropping the existing "geography" category, and one of the onehotcoded columns.

x = x.drop(['Geography', 0], axis = 1)
x.head()


# In[ ]:


#train and test set split, and feature scaling

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ### 2. Build the model using Keras
# ANN with many hidden layers is one of the branches of Deep Learning. 

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense #to add layers

#there is no rule on how many nodes each hidden layer should have
classifier = Sequential()
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
#init --> initialize weights according to uniform distribution
#input_dim is required for the first hidden layer, as it is the first starting point. --> number of nodes.
#output_dim --> number of nodes of the hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
#input_dim --> remove it as it already knows what to expect.

#the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#output_dim should be 1, as output is binary outcome, and activation should be 'sigmoid'
#If dependent variables have more than two categories, use activation = 'softmax'

#compile the model --> backpropagation -> gradient descent
classifier.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics = ['accuracy'])
#optimizer = algorithm to find the optimal set of weights in ANN
#loss = functions that should be optimized. if more than two categories, use "categorical_crossentropy"
#metrics = criterion used to calculate the performance of the model.


# Now let's run the model.

# In[ ]:


classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 20)
#batch_size = the number of observations after which you want to update the weights
#           batch size and epochs should be tuned through experiments.
#epoch = going through the whole dataset


# In[ ]:


#predicting the results

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #to classify each probability into True or False

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print (cm, '\n\n', y_pred[:5, :])


# In[ ]:


#accuracy
print ((1548 + 139)/2000)


# The fact that accuracy on train and test set are similar shows that the model did not overfit on the train set. Hyperparameters can be tuned to obtain better results.
