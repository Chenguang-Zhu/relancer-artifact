#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import keras

from matplotlib import pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (15, 8)

# Input data files are available in the "../../../input/barelydedicated_bank-customer-churn-modeling/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
print(os.listdir("../../../input/barelydedicated_bank-customer-churn-modeling"))


# In[ ]:


#importing the dataset
bank_df = pd.read_csv("../../../input/barelydedicated_bank-customer-churn-modeling/Churn_Modelling.csv", index_col='RowNumber')
bank_df.head()


# In[ ]:


bank_df.describe().T


# In[ ]:


bank_df.info()


# In[ ]:


## Removing surname as onhot encoding will cause issues for each one of them
bank_df=bank_df.drop(['CustomerId','Surname'], axis=1)


# In[ ]:


bank_df.info()    ## now we have 2 columns that have Object type.
                  ## Geography,Gender are 2 categorical variables, we need to encode them to make them numerical.


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[ ]:


## before splitting Data set into test and train, let us label encode the Gender and Geography.
bank_df.Gender.unique()   


# In[ ]:


bank_df.Gender=le.fit_transform(bank_df.Gender)
bank_df.Gender.unique()   


# In[ ]:


bank_df.Geography.unique()


# In[ ]:


bank_df.Geography=le.fit_transform(bank_df.Geography)
bank_df.Geography.unique()


# In[ ]:


bank_df.head()


# In[ ]:


X = np.array(bank_df.drop("Exited", axis=1))  ## defining X, Feature set


# In[ ]:


y = np.array(bank_df["Exited"])   ## defining y, target Varible
bank_df.Exited.unique()


# In[ ]:


y = keras.utils.to_categorical(y, num_classes=2)  #Encoding the output class label (One-Hot Encoding)
y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
test_size = 0.30 # taking 70:30 training and test set
seed = 7  # Random numbmer seeding for reapeatability of the code
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[ ]:


print("Dimension of train samples:",x_train.shape)
print("Dimension of test samples:",x_test.shape)
print("Dimension of train samples:",y_train.shape)
print("Dimension of test samples:",y_test.shape)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.utils import np_utils
import pickle

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Define model
model = Sequential()

model.add(keras.layers.Dense(output_dim = 20, init = 'uniform', activation = 'relu', input_dim = 10))
# Normalize the data
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(output_dim = 10, init = 'uniform', activation = 'relu'))
model.add(keras.layers.Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
model.add(keras.layers.Dense(output_dim = 2, init = 'uniform', activation = 'softmax'))

# Loss and Optimizer
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


# Train the model
history=model.fit(x_train, y_train, batch_size = 25, nb_epoch = 100,validation_data=(x_test, y_test))


# In[ ]:


y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)   


# In[ ]:


y_pred[0:5]


# In[ ]:


y_pred = y_pred.astype('float32')


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(cm)


# In[ ]:


print (((cm[0][0]+cm[1][1])*100)/(len(y_test)), '% of testing data was classified correctly')


# In[ ]:


score_train = model.evaluate(x_train, y_train,verbose=0)
print("Train Loss=",score_train[0],"; Accuracy=",score_train[1])
score = model.evaluate(x_test, y_test,verbose=0)
print("Test Loss=",score[0],"; Accuracy=",score[1])


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(np.array(history.history['accuracy']) * 100)
plt.plot(np.array(history.history['val_accuracy']) * 100)
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'validation'])
plt.title('Accuracy over epochs')
print()


# In[ ]:




