#!/usr/bin/env python
# coding: utf-8

# # About Dataset
# 
# ## Context
# This is the dataset used in the section "ANN (Artificial Neural Networks)" of the Udemy course from Kirill Eremenko (Data Scientist & Forex Systems Expert) and Hadelin de Ponteves (Data Scientist), called Deep Learning A-Z™: Hands-On Artificial Neural Networks. The dataset is very useful for beginners of Machine Learning, and a simple playground where to compare several techniques/skills.
# 
# It can be freely downloaded here: https://www.superdatascience.com/deep-learning/
# 
# The story: A bank is investigating a very high rate of customer leaving the bank. Here is a 10.000 records dataset to investigate and predict which of the customers are more likely to leave the bank soon.
# 
# ## Acknowledgements
# Udemy instructors Kirill Eremenko (Data Scientist & Forex Systems Expert) and Hadelin de Ponteves (Data Scientist), and their efforts to provide this dataset to their students.
# 
# # Solution
# This problem is solved by using an Artificial Nueral Network (ANN). The instructors of __Deep Learning A-Z™: Hands-On Artificial Neural Networks__ set the milestone for the accuracies as follows:
# * 84% Accuracy = Bronze
# * 85% Accuracy = Silver
# * 86%+ Accuracy = Gold
# 
# I tired different topologies for the ANN (Deeper Topology vs Wider Topology. [Read More](https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/)) with the combinations of different optimizers, dropouts etc. and found out that the deeper architecture with no dropout resulted in __86.8%__ Accuracy. Following is my code.

# Importing Dataset and splitting features and label

# In[ ]:


import numpy as np
import pandas as pd


from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)


dataset = pd.read_csv("../../../input/filippoo_deep-learning-az-ann/Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical features which are __sex__ & __Geography__. We will also do one hot encoding for __Geography__ to avoid dummy variable trap

# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()


# Splitting into training and testing sets

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Applying features scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Defining architecture for our neural network

# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense


def deep_model():
    classifier = Sequential()
    classifier.add(Dense(units=12, kernel_initializer='uniform', activation='relu', input_dim=12))
    classifier.add(Dense(units=12, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


# Running our ANN

# In[ ]:


from keras.models import load_model

# classifier = load_model('Churn_Modelling.h5')


# In[ ]:


# Uncomment to train new model otherwise it will used trained model
classifier = deep_model()
classifier.fit(X_train, y_train, batch_size=4, epochs=128)


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print("Accuracy: "+ str(accuracy*100)+"%")


# In[ ]:


classifier.save('Churn_Modelling.h5')


# In[ ]:




