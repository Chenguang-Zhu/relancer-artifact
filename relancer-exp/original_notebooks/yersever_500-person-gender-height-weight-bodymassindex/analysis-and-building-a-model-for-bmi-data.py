#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
print()
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

import os
print(os.listdir("../../../input/yersever_500-person-gender-height-weight-bodymassindex"))


# **Importing Data**

# In[ ]:


data = pd.read_csv(u"../../../input/yersever_500-person-gender-height-weight-bodymassindex/500_Person_Gender_Height_Weight_Index.csv")
data.head()


# **Checking for Missing Values(Any thin horizontal white lines in this image means a data is missing).**No missing values hence we are good to proceed 

# In[ ]:


import missingno as msno
msno.matrix(data)


# **Converting Weight from retarded units(lbs) to kg and visualising a scatterplot**

# In[ ]:


data['Weight'] = data['Weight'].apply(lambda x:0.453592*x)
data.plot.scatter(x='Weight', y='Height')


# **A hexbin density based plot would be more intuitive than scatterplot due high overlapping of data**

# In[ ]:


#data.plot.hexbin(x="Weight", y='Height',gridsize=15)
import seaborn as sns
sns.jointplot(x='Weight', y='Height', data=data, kind='hex', gridsize=15)


# **Amount of people who are  0 - Extremely Weak, 1 - Weak, 2 - Normal, 3 - Overweight, 4 - Obesity, 5 - Extreme Obesity**

# In[ ]:


#Index Reference : 0 - Extremely Weak, 1 - Weak, 2 - Normal, 3 - Overweight, 4 - Obesity, 5 - Extreme Obesity
data['Index'].value_counts().sort_index().plot.bar( figsize=(12, 6), fontsize=16, title='Frequency of Body Height-Weight Index', stacked=True ) 


# In[ ]:


data.info()


# **Categorising data as 1 for male and 0 for female(Not trying to be offensive :P)**

# In[ ]:


data['Gender'].replace({'Male':1,'Female':0},inplace = True)
data.head()


# Now moving onto Classification models we will try 4 basic methods of classification:
# 1. K-NN Classification.
# 2. Naive Bayes
# 3. Random Forest
# 4. XGB Classfier
# After classification we will compare all the models performance from it's confusion matrix and through k-fold crosss validation method.

# **Preparing and splitting data for training and testing**

# In[ ]:


X = data.iloc[:, [0,1,2]].values
y = data.iloc[:, 3].values


# **The data will have 400 training data and 100 testing data**

# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Trying out the K-Nearest Neighbour classifier model and visualising the confusion matrix.
# 
# **K-NN gives 86 % right prediction on the test set .**
# 
# **The k-fold cross validation removes the data bias and gives an accuarcy of 86.51% ± 3.63% **

# In[ ]:


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# In[ ]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
correct = cm.diagonal().sum()/cm.sum()
print("% of right predictions {0}".format(correct*100))
# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())
import numpy as np
import matplotlib.pyplot as plt

conf_arr = cm

norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, interpolation='nearest') 

width, height = conf_arr.shape

for x in range(width):
    for y in range(height):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center') 

print()
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
plt.xticks(range(width), alphabet[:width])
plt.yticks(range(height), alphabet[:height])
plt.savefig('confusion_matrix.png', format='png')


# Trying out the Naive Bayes classifier model and visualising the confusion matrix.
# 
# **Naive Bayes gives 67 % right prediction on the test set .**
# 
# **The k-fold cross validation  gives an accuarcy of 71.54% ± 6.05% **

# In[ ]:


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[ ]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
correct = cm.diagonal().sum()/cm.sum()
print("% of right predictions {0}".format(correct*100))
# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())
conf_arr = cm

norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, interpolation='nearest') 

width, height = conf_arr.shape

for x in range(width):
    for y in range(height):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center') 

print()
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
plt.xticks(range(width), alphabet[:width])
plt.yticks(range(height), alphabet[:height])
plt.savefig('confusion_matrix.png', format='png')


# Trying out the Random Forest classifier model and visualising the confusion matrix.
# 
# **Random Forest Classifier gives 89 % right prediction on the test set .**
# 
# **The k-fold cross validation gives an accuarcy of 84.78% ± 4.82% **

# In[ ]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)


# In[ ]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
correct = cm.diagonal().sum()/cm.sum()
print("% of right predictions {0}".format(correct*100))
# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())
conf_arr = cm

norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, interpolation='nearest') 

width, height = conf_arr.shape

for x in range(width):
    for y in range(height):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center') 

print()
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
plt.xticks(range(width), alphabet[:width])
plt.yticks(range(height), alphabet[:height])
plt.savefig('confusion_matrix.png', format='png')


# Trying out the XGB Classifier model and visualising the confusion matrix.
# 
# **XGB Classifier gives 82 % right prediction on the test set .**
# 
# **The k-fold cross validation gives an accuarcy of 84.90% ± 4.75% **

# In[ ]:


# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


# In[ ]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
correct = cm.diagonal().sum()/cm.sum()
print("% of right predictions {0}".format(correct*100))
# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())
conf_arr = cm

norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, interpolation='nearest') 

width, height = conf_arr.shape

for x in range(width):
    for y in range(height):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center') 

print()
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
plt.xticks(range(width), alphabet[:width])
plt.yticks(range(height), alphabet[:height])
plt.savefig('confusion_matrix.png', format='png')


# Now it is upto us to choose the model which is most suited.
# 
# Both Random Forest and XGB Classifiers are well suited.
# 
# I would go wtih Random forest as I find it more intuitive.

# In[ ]:




