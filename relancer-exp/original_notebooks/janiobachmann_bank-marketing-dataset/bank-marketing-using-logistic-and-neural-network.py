#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## This is my first kernel, any comment is very appreciatable. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Importing dataset 
data = pd.read_csv("../../../input/janiobachmann_bank-marketing-dataset/bank.csv")
#Reading the only 5 rows of the dataset
data.head()


# In[ ]:


#Describe function gives the basic stats value of the dataset
'''Their are no missing values in the data, if any fill them with mean, median or mode.'''

data.describe()


# In[ ]:


##Ploting the dataset using seaborn which is one of the best ways to visualize your data
## I hvae just showed different seaborn plots , you can play with them different variables
sns.set(style="ticks", color_codes=True)
##scatter plot
sns.stripplot(x="job", y="duration", data=data)
sns.catplot(x="job", y="duration", kind = "swarm", data=data)


# In[ ]:


##Distribution plot


# In[ ]:


sns.boxplot(x="job", y="duration", data=data)


# In[ ]:


sns.violinplot(x="job", y="duration", data=data)


# In[ ]:


sns.boxenplot(x="contact", y="age", data=data)


# In[ ]:


##estimate plot
sns.pointplot(x="job", y="duration", data=data)


# In[ ]:


sns.barplot(x="age", y="job", data=data)


# In[ ]:


data['age'].describe()


# In[ ]:


data.columns


# In[ ]:


## Checking for catogorical variable 
data['education'].unique()


# In[ ]:


data['marital'].unique()


# In[ ]:


##Ploting the count of marital and education
values = data['marital'].value_counts().tolist()
labels = ['married', 'divorced', 'single']


# In[ ]:


val = data['education'].value_counts().tolist()
lab = ['secondary', 'tertiary', 'primary', 'unknown']


# In[ ]:


sns.barplot(x=labels, y=values, data=data)
plt.title("Count by Marital Status")
print()


# In[ ]:


sns.barplot(x=lab, y=val, data=data)
plt.title("Count by Education Status")
print()


# In[ ]:


##Pair ploting 

sns.set(style="ticks")
print()
print()


# In[ ]:


## Now seperate the dependant and inpedepandant variable before proceeding

X = data.iloc[:, 0:16]
Y = data.iloc[:, 16]


# In[ ]:


X


# In[ ]:


Y


# In[ ]:


#Splitting the catagorical variable to 0s and 1s using LabelEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder().fit_transform(['no','yes'])
housing = LabelEncoder().fit_transform(X['housing'])
loan = LabelEncoder().fit_transform(X['loan'])
deposit = LabelEncoder().fit_transform(Y)
job = LabelEncoder().fit_transform(X['job'])
marital = LabelEncoder().fit_transform(X['marital'])
education = LabelEncoder().fit_transform(X['education'])
poutcome = LabelEncoder().fit_transform(X['poutcome'])


# In[ ]:


X['housing'] = housing
X['loan']= loan
X['job'] = job
X['marital']=marital
X['education']=education
X['poutcome']=poutcome
Y=deposit
del X['month']
del X['day']
del X['contact']
del X['default']



# In[ ]:


X


# In[ ]:


Y


# In[ ]:


X.shape


# In[ ]:


Y.shape


# In[ ]:


onehotencoder = OneHotEncoder(categorical_features = [1,2,3,11])
X = onehotencoder.fit_transform(X).toarray()


# In[ ]:


X


# In[ ]:


X.shape


# In[ ]:


## PLoting housing vs deposit shows us a clear view how many people have deposited interms of hosuing 
sns.barplot(x=deposit, y=housing, data= data)
plt.xlabel("number of people deposit")
plt.ylabel("Housing")
plt.title("Deposit vs Housing")
print()


# In[ ]:


### This plot bet loan vs deposit has shown 83% has not deposited and 17% remaining have deposits 
#which gives a weight to output. 
sns.barplot(x=deposit, y=loan, data= data)
plt.xlabel("number of people deposit")
plt.ylabel("Loans Taken")
plt.title("Deposit vs Loans")
print()


# In[ ]:


## spliting the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 123)


# In[ ]:


##Feature scaling 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


X_train


# In[ ]:


X_test


# In[ ]:


### for model selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier


# In[ ]:


svm = SVC()
knn = KNeighborsClassifier(n_neighbors=5)
lr = LogisticRegression()
grad_clf = GradientBoostingClassifier()
tree_clf = tree.DecisionTreeClassifier()
neural_clf = MLPClassifier(alpha=1)


# In[ ]:


from sklearn.model_selection import cross_val_score
log_scores = cross_val_score(lr, X_train, y_train, cv=3)
log_reg_mean = log_scores.mean()


# In[ ]:


log_reg_mean


# In[ ]:


knn_scores = cross_val_score(knn, X_train, y_train, cv=3)
knn_mean = knn_scores.mean()


# In[ ]:


knn_mean


# In[ ]:


grad_scores = cross_val_score(grad_clf, X_train, y_train, cv=3)
grad_mean = grad_scores.mean()


# In[ ]:


grad_mean


# In[ ]:


tree_scores = cross_val_score(tree_clf, X_train, y_train, cv=3)
tree_mean = tree_scores.mean()


# In[ ]:


tree_mean


# In[ ]:


neural_scores = cross_val_score(neural_clf, X_train, y_train, cv=3)
neural_mean = neural_scores.mean()


# In[ ]:


neural_mean


# In[ ]:


##Stocastic gradient decent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(n_iter=10000, alpha=0.03)
sgd.fit(X_train,y_train)

sgd.score(X_test,y_test)


# In[ ]:


from sklearn.model_selection import cross_val_predict

cross_val_predict(grad_clf,X_test,y_test).mean()


# In[ ]:


##Fitting the model to the test set 
knn.fit(X_train,y_train)

knn.score(X_test,y_test)


# In[ ]:


svm.fit(X_train,y_train)
svm.score(X_test,y_test)


# In[ ]:


### Creating a confusion matrix
from sklearn.metrics import confusion_matrix

y_pred = cross_val_predict(neural_clf,X_test,y_test,cv=4)
conf = confusion_matrix(y_test,y_pred)


# In[ ]:


conf


# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))
print()
plt.title("Confusion Matrix", fontsize=20)
plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.99)
ax.set_yticks(np.arange(conf.shape[0]) + 0.5, minor=False)
ax.set_xticklabels("")
ax.set_yticklabels(['Refused T. Deposits', 'Accepted T. Deposits'], fontsize=16, rotation=360)
print()


# In[ ]:


### Its very important to check recall, precision, accuracy and F1 score
from sklearn.metrics import precision_score,recall_score,f1_score


recall_score(y_test,y_pred)


# In[ ]:


precision_score(y_test,y_pred)


# In[ ]:


f1_score(y_test,y_pred)


# In[ ]:


## XG boosting is very good model to boost your model accuracy 
##Boostig algorithm XGBoost
import xgboost as xgb

cl = xgb.XGBClassifier(max_depth=4,n_estimators=20)

cl.fit(X_train,y_train)

y_xg = cl.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score,precision_recall_fscore_support

accuracy_score(y_test,y_xg)

