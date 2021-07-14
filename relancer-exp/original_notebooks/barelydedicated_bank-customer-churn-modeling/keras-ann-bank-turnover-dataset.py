#!/usr/bin/env python
# coding: utf-8

#   # keras ANN - Bank Turnover Dataset
#   
#   # Importng the libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


# # Imporing Data

# In[ ]:


dataset = pd.read_csv("../../../input/barelydedicated_bank-customer-churn-modeling/Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values


# # Encoding categorical data

# In[ ]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)

labelencoder_x_2=LabelEncoder()
X[:,4]=labelencoder_x_2.fit_transform(X[:,4])

X=X[:, 1:]


# In[ ]:


X


# # splitting the dataset into the Training set and Test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)


# # Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train.reshape(-1,1))


# # Importing the Keras Libaraies and packages

# In[ ]:


import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# # ANN Network

# In[ ]:


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and first hidden layer with Dropout
classifier.add(Dense(output_dim = 6,init = 'uniform', activation = 'relu', input_dim=11))
classifier.add(Dropout(p = 0.1))

# Adding Second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu' ))# init = 'uniform'==> init weight randomly, activation Function = 'relu' 
classifier.add(Dropout(p = 0.1))

# Adding ouyput layer
classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid' ))

# Compiing the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# # Fitting the ANN to the Training set

# In[ ]:


classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


# # Predicting using the training set
# 

# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


y_pred = (y_pred > 0.5)


# # Making a single Prediction
# * Geography: France 
# * Credit Score: 600
# * Gender: Male
# * Age: 40 years old
# * Tenure: 3 years
# * Balance: $60000
# 
# * Number of Products: 2
# * Does this customer have a credit card ? Yes
# * Is this customer an Active Member: Yes
# * Estimated Salary: $50000
# * So should we say goodbye to that customer ?

# In[ ]:


new_pred = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,5000]])))
new_pred = (new_pred > 0.5)


# # Making the confusion matrix**

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# # Evaluating our ANN

# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6,init = 'uniform', activation = 'relu', input_dim=11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu' ))# init = 'uniform'==> init weight randomly, activation Function = 'relu' 
    classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid' ))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


# In[ ]:


classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, nb_epoch=100)


# In[ ]:


accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1) 


# In[ ]:


mean = accuracies.mean()


# In[ ]:


variance = accuracies.var()


#    # Improving the ANN
#    # Dropout regularization to reduce overfitting if needed

# # Tuning the ANN

# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer = 'adam'):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6,init = 'uniform', activation = 'relu', input_dim=11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu')) 
    classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

# parameters = {'batch_size': range(25, 32),
#               'epochs': range(100, 501),
#               'optimizer': ['adam', 'rmsprop']}

parameters = {'batch_size': [25, 32], 'epochs': [100, 103], 'optimizer': ['adam', 'rmsprop']} 

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10) 


# In[ ]:


grid_search = grid_search.fit(X_train,y_train)


# In[ ]:


# # summarize results
best_param = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best accuracy: %f\nusing parameters : %s" % (best_accuracy,best_param))



# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
# 	print("%f (%f) with: %r" % (mean, stdev, param))
    

