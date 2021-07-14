#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print()
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.preprocessing import normalize, Normalizer # data normalizers
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# path to save figures
if not os.path.exists('./Figures'):
    os.mkdir('./Figures')


# ## Data Exploration

# In[ ]:


# reading the data using pandas routine
data = pd.read_csv("../../../input/adammaus_predicting-churn-for-bank-customers/Churn_Modelling.csv")

#printing 5 first rows od the data
data.head()


# From the table above; We can drop the first three columns as neither RowNUmber nor customerId or the name of the customer contribute to the fact that a customer can exit the bank or not. 

# In[ ]:


# general information about the data
data.info()


# From the above we realize that we have 3 categorical data. We can drop one of them(Surname) as it will not contribute to our target feature(Exited) during this work. 

# In[ ]:


# Drop the irrelevant columns  as shown above
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)


# In[ ]:


# One-Hot encoding our categorical attributes
"""As Gender will gibve us two columns we can drop male column and renaim the Gender_Female to Gender such that 1 corresponds to the female customer while 0 corresponds to male customer""" 
cat_attr = ['Geography', 'Gender']
data = pd.get_dummies(data, columns = cat_attr, prefix = cat_attr)

data = data.drop(["Gender_Male"], axis = 1)
data.rename(columns={"Gender_Female": "Gender(1:F,0:M)"}, inplace=True)


# In[ ]:


data.head()


# In[ ]:


# five point descriptive statistics, and std
data.describe()


# From the above; We realize that there is no missing data from the count row.

# In[ ]:


# Checking for unique value in the data attributes
data.nunique()


# In[ ]:


# Now lets check the class distributions
sns.countplot("Exited",data=data)
plt.title('Histogram')
plt.savefig('./Figures/Class distribution')


# In[ ]:


# Now lets check the class distributions
sns.countplot("Gender(1:F,0:M)",data=data)
plt.title('Histogram')
plt.savefig('./Figures/Gender distribution')


# In[ ]:


"""Data are unmbalanced from the histogram. let us chetch the class percentage"""
N,D = data.shape
no_exited_pct = np.sum(data.iloc[:,-5] == 0)/N
exited_pct = np.sum(data.iloc[:,-5] == 1)/N

print('No exited customer  Class precentage: {:2f}'.format(100*no_exited_pct))
print('Exited customer Class precentage: {:2f}'.format(100*exited_pct))




# In[ ]:


# group the data by the target variable 
data_grp= data.groupby('Exited')


# In[ ]:


# seeing the count per Exited customers data
data_grp.count()


# In[ ]:


# general describtive per Exited customers data
data_grp.describe()


# In[ ]:


# general data correlation
data.corr()


# In[ ]:



# general data correlation heatmap
print()
plt.title('Correlation')
plt.savefig('./Figures/corr_matrix_plot')


# In[ ]:


#arranging features with are highly correlated with the target (ascending order)
cor = data.corr()
corr_t = (cor ["Exited"]).abs()

print("The features which are most correlated with Exited feature:\n", corr_t.sort_values(ascending=False)[1:14].index)


# In[ ]:


#plotting all features 
print()
print()
plt.title('Pair plot')
plt.legend(['Not churn', 'churn'])
plt.savefig('./Figures/All in one plot')


# In[ ]:


#  churned customers class distributions on gender
churn     = data[data["Exited"] == 1]
not_churn = data[data["Exited"] == 0]
sns.countplot("Gender(1:F,0:M)",data=churn)
plt.title('Histogram_churn_Customers')
plt.savefig('./Figures/Gender_churn distribution')


# In[ ]:


#  not churned customers class distributions on gender
churn     = data[data["Exited"] == 1]
not_churn = data[data["Exited"] == 0]
sns.countplot("Gender(1:F,0:M)",data=not_churn)
plt.title('Histogram_churn_Customers')
plt.savefig('./Figures/Gender_not_churn distribution')


# ## Dimentionality reduction
# The data we have is highly nonlinear as shown by the scater plots from pair plot. We are considering the dimentionality reduction as a preprocessig and feature engenering step to better represent the data.

# In[ ]:


# dimentionality reduction

pca = PCA(2) # to get the independent representation
tsne = TSNE(2) # state of the art / the data is nonlinear
iso = Isomap() # follwing the assumption that the data lives in nonlinear manifold.


# In[ ]:


# fit the dimensionality reduction algorithm 

pca_data = pca.fit_transform(normalize(data))
tsne_data = tsne.fit_transform(normalize(data))
iso_data = iso.fit_transform(normalize(data))


# In[ ]:


# ploting 
f, (ax1, ax2, ax3) = plt.subplots(3, 1,)


ax1.set_title('PCA')
ax1.scatter(pca_data[:,0],pca_data[:,1], c = data['Exited'])
plt.title('PCA plot')


ax2.set_title('TSNE')
ax2.scatter(tsne_data[:,0],tsne_data[:,1], c = data['Exited'])
plt.title('TSNE')


ax3.set_title('Isomap')
ax3.scatter(iso_data[:,0],iso_data[:,1], c = data['Exited'])
plt.title('ISOMAP')
plt.savefig('./Figures/PCA,TSNE,ISOMAP')


# # Splitting the data

# In[ ]:


from sklearn.model_selection import train_test_split

X = data.drop(["Exited"],axis=1)
y = data.Exited
train_data,test_data, target_train, target_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


train_data.head()


# In[ ]:


print(len(train_data))
print(len(test_data))


# ### Evaluation criteria
# As the data is unbalance we are going to look at confusion matrix values like recal and f1 score which is the best way to judge the model from unbalanced data as we can not totally relay on the testing accuracy.

# # 1.1- k-Nearest Neighbors (k-NN) Algorithm

# In[ ]:



np.random.seed(6)
X=normalize(train_data)
y=target_train
# search for an optimal value of K for KNN with cross validation
# range of k we want to try
k_range = range(2, 39)
# empty list to store scores
k_scores = []
# 1. we will loop through reasonable values of k
for k in k_range:
    # 2. run KNeighborsClassifier with k neighbours
    knn = KNeighborsClassifier(n_neighbors=k)
    # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
    scores = cross_val_score(knn, X, y, cv = 10, scoring='accuracy')
    #scores = cross_val_score(knn, inputs, label, cv = 10, scoring='accuracy')
    # 4. append mean of scores for k neighbors to k_scores list
    k_scores.append(scores.mean())

print("k different averages:", k_scores)
print("\nMax average, index of Max:", max(k_scores),"||", k_scores.index(max(k_scores)))
pos = k_scores.index(max(k_scores))


# In[ ]:


# plot how accuracy changes as we vary k
# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated accuracy')
plt.title('Cross-Validate Accuracy Vs K')
plt.savefig('./Figures/Cross-Validate Accuracy Vs K')


# In[ ]:



model = KNeighborsClassifier(3)
print("\t %%% Now lets fit our model first and then we test with our test set %%%")
model = model.fit(X,y)
print("\n Training score: ",model.score(X, y)) 
pred = model.predict(normalize(test_data))
score = metrics.accuracy_score(pred, target_test)
print("\nThe accuracy score that we get is: ",score)  
print("\n Confusion Matrix: ", confusion_matrix(target_test, pred))
print(metrics.classification_report(target_test, pred))


# # 1.2- Extra trees Classifier

# In[ ]:



ETC = ExtraTreesClassifier(random_state=5)
ETC = ETC.fit(X, y)
print("\n Training score: ",ETC.score(X, y)) #evaluating the training error
pred = ETC.predict(normalize(test_data))
score = metrics.accuracy_score(pred,target_test)
print("\nThe accuracy score that we get is: ",score)
print("\n Confusion Matrix: ", confusion_matrix(target_test, pred))
print(metrics.classification_report(target_test, pred))


# #  1.3- Random Forest Classifier

# In[ ]:


input_train, label_input = X, y
input_test, label_test = normalize(test_data), target_test

RF = RandomForestClassifier(max_depth=11, random_state=5)
RF = RF.fit(input_train, label_input)
print("\n Training score: ",RF.score(input_train, label_input)) #evaluating the training error
pred = RF.predict(input_test)
score = metrics.accuracy_score(pred,label_test)
print("\nThe accuracy score that we get is: ",score)
print("\n Confusion Matrix: ", confusion_matrix(label_test, pred))
print(metrics.classification_report(label_test, pred))


# In[ ]:


# getting featrure importance
RF.feature_importances_


# # 1.4- Dummy Classifier

# In[ ]:



dc = DummyClassifier(strategy="uniform",random_state=5)
dc = dc.fit(input_train, label_input)
print("\n Training score: ",dc.score(X, y)) #evaluating the training error
pred = dc.predict(input_test)
score = metrics.accuracy_score(pred,label_test)
print("\nThe accuracy score that we get is: ",score)
print("\n Confusion Matrix: ", confusion_matrix(label_test, pred))
print(metrics.classification_report(label_test, pred))


# # 1.5- Ada Boost Classifier

# In[ ]:



ABC = AdaBoostClassifier(random_state=5)
ABC = ABC.fit(input_train, label_input)
print("\n Training score: ",ABC.score(input_train, label_input)) #evaluating the training error
pred = ABC.predict(input_test)
score = metrics.accuracy_score(pred,label_test)
print("\nThe accuracy score that we get is: ",score)
print("\n Confusion Matrix: ", confusion_matrix(label_test, pred))
print(metrics.classification_report(label_test, pred))


# # 1.6- Gradient Boosting Classifier

# In[ ]:



GBC = GradientBoostingClassifier(random_state=5)
GBC = GBC.fit(input_train, label_input)
print("\n Training score: ",GBC.score(input_train, label_input)) #evaluating the training error
pred = GBC.predict(input_test)
score = metrics.accuracy_score(pred,label_test)
print("\nThe accuracy score that we get is: ",score)
print("\n Confusion Matrix: ", confusion_matrix(label_test, pred))
print(metrics.classification_report(label_test, pred))


# # 1.7- Decision Tree Classifier

# In[ ]:



dt = tree.DecisionTreeClassifier(random_state=5)
dt = dt.fit(input_train, label_input)
print("\n Training score: ",dt.score(input_train, label_input)) #evaluating the training error
pred = dt.predict(input_test)
score = metrics.accuracy_score(pred,label_test)
print("\nThe accuracy score that we get is: ",score)
print("\n Confusion Matrix: ", confusion_matrix(label_test, pred))
print(metrics.classification_report(label_test, pred))


# # 1.8 - SGD Classifier

# In[ ]:



sgdc = SGDClassifier(loss="hinge", penalty="l2", max_iter=25,random_state=5)
sgdc = sgdc.fit(input_train, label_input)
print("\n Training score: ",sgdc.score(input_train, label_input)) #evaluating the training error
pred = sgdc.predict(input_test)
score = metrics.accuracy_score(pred,label_test)
print("\nThe accuracy score that we get is: ",score)
print("\n Confusion Matrix: ", confusion_matrix(label_test, pred))
print(metrics.classification_report(label_test, pred))


#   # 1.9 -  Train a binary classifier to predict if the customer will churn or not
# 

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler


# In[ ]:



#Scaling data
scaler = MinMaxScaler()
scaler.fit(X)
dfx = scaler.transform(X)
scaler.fit(test_data)
dfx_test = scaler.transform(test_data)


# In[ ]:


#Neural network

model0 = keras.Sequential([ keras.layers.Flatten(input_shape=dfx.shape[1:]), keras.layers.Dense(128, activation=tf.nn.relu), keras.layers.Dense(1, activation=tf.nn.sigmoid) ]) 


model0.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']); 


model0.fit(dfx, y, epochs=100);


# In[ ]:


test_loss, test_acc = model0.evaluate(dfx_test, target_test)

print('Test accuracy:', test_acc)


y_hat = model0.predict_classes(dfx_test)

print(metrics.classification_report(target_test, y_hat))


# ##  Using Sampling methods to overcome the unbalance in the dataset

# In[ ]:


# helper ploting function 

def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter( X[y==l, 0], X[y==l, 1], c=c, label=l, marker=m ) 
    plt.title(label)
    plt.legend(loc='upper right')
    print()


# # balancing the data

# In[ ]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy = 'auto', kind = 'regular',random_state=5)

data1 = data.drop(["Exited"],axis=1)
inputs,label = sm.fit_sample(data1, data['Exited'])
print("Original dataset: ",data['Exited'].value_counts())


compt = 0
for i in range(len(label)):
    if label[i]==1:
        compt += 1
print("\nNumber of 1 in the new  dataset: ",compt)


# In[ ]:


plot_2d_space(inputs, label, 'balanced data scatter plot')
plt.savefig("Scatter plot balanced data")



# In[ ]:


Xc = inputs
yc = label
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)


# # 2 - Modelling with balanced data

# In[ ]:


print('-----------------Gradient Boost Classifier--------------------')
GBC = GradientBoostingClassifier(random_state=5)
GBC = GBC.fit(Xc_train, yc_train)
pred = GBC.predict(Xc_test)
score = metrics.accuracy_score(pred,(yc_test))
print("Accuracy---", accuracy_score(yc_test,pred))
print(metrics.classification_report(yc_test, pred))


print('-----------------Ada Boost Classifier--------------------')
ABC = AdaBoostClassifier(random_state=5)
ABC = ABC.fit(Xc_train, yc_train)
pred = ABC.predict(Xc_test)
print("Accuracy---", accuracy_score(yc_test,pred))
print(classification_report(yc_test,pred))


print('-----------------RandomForestClassifier--------------------')
model  = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0).fit(Xc_train, yc_train)
pred = model.predict(Xc_test)
print("Accuracy---", accuracy_score(yc_test,pred))
print(classification_report(yc_test,pred))



# In[ ]:


#Try the neural network also
#Scaling data
scaler = MinMaxScaler()
scaler.fit(Xc_train)
dfx = scaler.transform(Xc_train)
scaler.fit(Xc_test)
dfx_test = scaler.transform(Xc_test)


# In[ ]:


model0.fit(dfx, yc_train, epochs=100);


# In[ ]:


test_loss, test_acc = model0.evaluate(dfx_test, yc_test)

print('Test accuracy:', test_acc)


# In[ ]:


y_hat = model0.predict_classes(dfx_test)

print(metrics.classification_report(yc_test, y_hat))


# # Conclusion

# In[ ]:




