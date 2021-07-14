#!/usr/bin/env python
# coding: utf-8

# # Bank Churn Prediction : Grid Search VS Threshold modification

# # Introduction
# 
# The data set contains details of a bank's customers and the target variable is a binary variable reflecting the fact whether the customer left the bank (closed his account : Exited = 0) or he continues to be a customer.
# 
# The problem present here is a classification problem, or more specificaly binary classification problem. We can solve this kind of problems using famous models like Logistic Regression, Linear Discriminant Analysis, SVC, XGBoost, LGBM Classifier, Decision Tree Based models ...
# 
# The model we choose for deployment will be the one who 

# # 1. Import Libraries:

# In[ ]:


import pandas as pd
import numpy as np
import re
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
print()



import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier) 
from sklearn.svm import SVC
from sklearn.model_selection import KFold

from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


from sklearn.metrics import accuracy_score, recall_score,roc_curve, auc

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier


from mlxtend.classifier import EnsembleVoteClassifier



from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


# In[ ]:


pd.options.display.max_colwidth = 200


# # 2. Import Data

# In[ ]:


# Load in the dataset
DF = pd.read_csv("../../../input/shrutimechlearn_churn-modelling/Churn_Modelling.csv")
DF.head()


# # 3. EDA & Data Preprocessing:

# The variables in this dataset are represented as follow: 
# 
# * Surname : The surname of the customer
# * CreditScore : The credit score of the customer
# * Geography : The country of the customer(Germany/France/Spain)
# * Gender : The gender of the customer (Female/Male)
# * Age : The age of the customer
# * Tenure : The customer's number of years in the bank
# * Balance : The customer's account balance
# * NumOfProducts : The number of bank products that the customer uses
# * HasCrCard : Does the customer has a card? (0=No,1=Yes)
# * IsActiveMember : Does the customer has an active membership (0=No,1=Yes)
# * EstimatedSalary : The estimated salary of the customer
# * Exited : Churned or not? (0=No,1=Yes)

# In[ ]:


DF.info()


# 1. Types of our varibles:
# 
# Categorical: Exited, Gender,HasCrCard, IsActiveMember, and Geography.
# 
# Continous: Age, CreditScore, EstimatedSalary and Balance. Discrete: Tenure, NumOfProducts.
# 
# 2. Null and Missing values:
# 
# The data contains no Null values, so no deletion or imputation are needed.

# In[ ]:


DF["IsActiveMember"] = DF["IsActiveMember"].map({1:"Yes" , 0:"No"})
DF["Exited"] = DF["Exited"].map({1:"Yes" , 0:"No"})
DF["HasCrCard"] = DF["HasCrCard"].map({1:"Yes" , 0:"No"})


# In[ ]:


DF.iloc[:,3:len(DF)].describe([0.01,0.1,0.25,0.5,0.75,0.99])


# First thing we can observe from the summary statistics table, is that we have an umbalanced dataset. 20% of customers churned and about 80% don't. So, with this information, we had to change our metric for choosing the best medel, instead of using the accuracy, we will use the recall.
# 
# Recall,  also referred to as the true positive rate or sensitivity, is the True Positive Rate.
# 
# In other term, recall attempts to answer the following question: What proportion of actual positives was identified correctly? So our metric will be the proportion of customers the model predect them to churn among the total number of true churners.

# In[ ]:


DF.iloc[:,3:len(DF)].describe(include=['O'])


# In[ ]:


DF["IsActiveMember"] = DF["IsActiveMember"].map({"Yes":1 , "No":0}).astype(int)
DF["Exited"] = DF["Exited"].map({"Yes":1 , "No":0}).astype(int)
DF["HasCrCard"] = DF["HasCrCard"].map({"Yes":1 , "No":0}).astype(int)


# We can also conclude from the summary statistics that the age variable has some outliers with very high age (up to 92), and that few customers have more than 3 products.
# 
# Now let's move to some more exploratory analysis for each variable.
# 
# ### 1. CreditScore:

# In[ ]:


sns.distplot(DF["CreditScore"], label="Skewness : %.2f"%(DF["CreditScore"].skew()))
plt.legend(loc="best");


# In[ ]:


plot = sns.kdeplot(DF["CreditScore"][(DF["Exited"] == 0) & (DF["CreditScore"].notnull())], color="Red", shade = True)
plot = sns.kdeplot(DF["CreditScore"][(DF["Exited"] == 1) & (DF["CreditScore"].notnull())], ax =plot, color="Blue", shade= True)
plot.set_xlabel("Credit Score")
plot.set_ylabel("Frequency")
plot = plot.legend(["Not Churn","Churn"])


# As we can see, CreditScore distribution is bell shaped. So no transformation is needed.
# 
# 
# ### 2. Geography:
# 

# In[ ]:


plot = sns.barplot(x = "Geography" , y = "Exited" , data = DF , errwidth = 0)

plt.ylim(0,0.5)
plt.yticks([0.1,0.2,0.3,0.4])
for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points') 
    
plt.ylabel("Churn proportion")
plt.title("Percentage of churning by the geography of the customer");


# In[ ]:


plot=sns.countplot(DF["Geography"], hue=DF["Exited"])

plt.ylim(0,4700)

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points') 

plt.legend(loc="best");


# It seems like the customers from Germany are more likely to churn comparing the the ones from France or Spain.
# 
# ### 3. Gender:

# In[ ]:


plot = sns.barplot(x = "Gender" , y = "Exited" , data = DF , errwidth = 0)

plt.ylim(0,0.5)
plt.yticks([0.1,0.2,0.3,0.4])
for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points') 

plt.ylabel("Churn proportion")
plt.title("Percentage of churning by the gender of the customer");


# In[ ]:


plot=sns.countplot(DF["Gender"], hue=DF["Exited"])

plt.ylim(0,5000)

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points') 

plt.legend(loc="best");


# There is no significant diffirence in the churn proportion between male and female customers.
# 
# ### 4. Age:
# 

# In[ ]:


sns.distplot(DF["Age"], label="Skewness : %.2f"%(DF["Age"].skew()))
plt.legend(loc="best");


# The Age distribution is very skewed, so we will try to transform the Age variable with the log transformation to get better distribution.

# In[ ]:


DF["LogAge"] = np.log(DF["Age"])

sns.distplot(DF["LogAge"], label="Skewness : %.2f"%(DF["LogAge"].skew()))
plt.legend(loc="best");


# Now we got a bell curve distribution better than the original age distribution. So we will use the LogAge as new varible in the modelisation.

# In[ ]:


plot = sns.kdeplot(DF["Age"][(DF["Exited"] == 0) & (DF["Age"].notnull())], color="Red", shade = True)
plot = sns.kdeplot(DF["Age"][(DF["Exited"] == 1) & (DF["Age"].notnull())], ax =plot, color="Blue", shade= True)
plot.set_xlabel("Age")
plot.set_ylabel("Frequency")
plot = plot.legend(["Not Churn","Churn"])


# The young cutomers have less chance to churn comparing to old customers.
# 
# ### 5. Tenure:

# In[ ]:


plot = sns.barplot(x = "Tenure" , y = "Exited" , data = DF , errwidth = 0)

plt.ylim(0,0.5)
plt.yticks([0.1,0.2,0.3,0.4])

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points') 
    
plt.ylabel("Churn proportion")
plt.title("Percentage of churning by the customer's number of years in the bank");


# In[ ]:


sns.countplot(DF["Tenure"], hue=DF["Exited"])
plt.legend(loc="best");


# The distribution of Tenure is uniform, and there is no difference between the two target categories.
# 
# ### 6. Balance:

# In[ ]:


sns.distplot(DF["Balance"], label="Skewness : %.2f"%(DF["Balance"].skew()))
plt.legend(loc="best");


# In[ ]:


plot = sns.kdeplot(DF["Balance"][(DF["Exited"] == 0) & (DF["Balance"].notnull())], color="Red", shade = True)
plot = sns.kdeplot(DF["Balance"][(DF["Exited"] == 1) & (DF["Balance"].notnull())], ax =plot, color="Blue", shade= True)
plot.set_xlabel("Balance")
plot.set_ylabel("Frequency")
plot = plot.legend(["Not Churn","Churn"])


# The balance variable has an extrem amount of low values, maybe refers to student or non income customers. For the rest of values, the variable follow a bell curve distribution.
# 
# The cutomers with low balance have less chance to churn comparing to the ones with hight Balance.
# 
# ### 7. NumOfProducts

# In[ ]:


plot = sns.barplot(x = "NumOfProducts" , y = "Exited" , data = DF , errwidth = 0)

plt.ylim(0,1.2)
for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points') 
    
plt.ylabel("Churn proportion")
plt.title("Percentage of churning by the Number of Products");


# In[ ]:



plot = sns.countplot(DF["NumOfProducts"], hue=DF["Exited"])

plt.ylim(0,4700)

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points') 
    

plt.legend(loc="right");


# The Number Of Products seems to be an effective variable to detect the customers who have more chance to churn. The customers with 1 or 2 products have a greatter chance to not churn, and in the other hand, the customers with 3 or 4 products have bigger chance to churn.
# 
# ### 8. HasCrCard:

# In[ ]:



plot = sns.countplot(DF["HasCrCard"], hue=DF["Exited"])

plt.ylim(0,6200)

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points') 

plt.legend(loc="best");


# In[ ]:


plot = sns.barplot(x = "HasCrCard" , y = "Exited" , data = DF , errwidth = 0)

plt.ylim(0,0.5)
for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points') 
    
plt.ylabel("Churn proportion");


# There is no difference for churn variable between the customers who have a Credit Card or haven't.
# 
# ### 9. IsActiveMember:

# In[ ]:



plot = sns.countplot(DF["IsActiveMember"], hue=DF["Exited"])

plt.ylim(0,5500)

for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points') 

plt.legend(loc="best");


# In[ ]:


plot = sns.barplot(x = "IsActiveMember" , y = "Exited" , data = DF , errwidth = 0)

plt.ylim(0,0.5)
for p in plot.patches:
    plot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points') 
    
plt.ylabel("Churn proportion");


# For the customer who have an active membership the chance of churn is less comparing to customers with no active membership.
# 
# ### 10. EstimatedSalary:

# In[ ]:



sns.distplot(DF["EstimatedSalary"], label="Skewness : %.2f"%(DF["Balance"].skew()))
plt.legend(loc="best");


# In[ ]:


plot = sns.kdeplot(DF["EstimatedSalary"][(DF["Exited"] == 0) & (DF["EstimatedSalary"].notnull())], color="Red", shade = True)
plot = sns.kdeplot(DF["EstimatedSalary"][(DF["Exited"] == 1) & (DF["EstimatedSalary"].notnull())], ax =plot, color="Blue", shade= True)
plot.set_xlabel("Estimated Salary")
plot.set_ylabel("Frequency")
plot = plot.legend(["Not Churn","Churn"])


# The distribution of salary follows a uniform distribution for both categories.
# 
# ### 11. Exited:

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(12,5))
DF['Exited'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Exited')
ax[0].set_ylabel('')
sns.countplot('Exited',data=DF,ax=ax[1])
print()


# ## Correlations:

# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
print()


# There is no significant correlation between any two features in train dataset.
# 
# 
# ## Data Preprocessing:

# In[ ]:


Y = DF["Exited"]
X = DF.drop(['RowNumber', 'CustomerId', 'Surname' ,'Age' ,'Exited'], axis=1)


# ### Dummy variables for Gender & Geography:

# In[ ]:



X = pd.get_dummies(X, columns = ["Geography"],drop_first = True)
X = pd.get_dummies(X, columns = ["Gender"],drop_first = True)

X.head()


# ### Standardization:

# In[ ]:


X["Balance"] = (X["Balance"] - X["Balance"].mean())/X["Balance"].std()


# In[ ]:


X["EstimatedSalary"] = (X["EstimatedSalary"] - X["EstimatedSalary"].mean())/X["EstimatedSalary"].std()

X["CreditScore"] = (X["CreditScore"] - X["CreditScore"].mean())/X["CreditScore"].std()

X.head()



# # 4. Modeling:
# 
# Now we are ready to train a model and predict the required solution. There are a lot of predictive modelling algorithms to choose from. We must understand the type of problem and solution requirement to narrow down to a selected few models which we can evaluate. Our problem is a classification problem, we want to identify relationship between output (Churn or not) with other features or informations about customers. The most used models in our case can be set as follows:
# 
# * Logistic Regression
# * XGBoost
# * AdaBoost
# * Gradient Boosting
# * Linear Discriminant Analysis
# * Logistic Regression
# * KNN or k-Nearest Neighbors
# * Support Vector Machines
# * Naive Bayes classifier
# * Decision Tree
# * Random Forrest
# * Perceptron
# * Artificial neural network

# In[ ]:


RS=121


classifiers = [ KNeighborsClassifier(), SVC(random_state=RS, probability = True), DecisionTreeClassifier(random_state=RS), RandomForestClassifier(random_state=RS), AdaBoostClassifier(random_state=RS), GradientBoostingClassifier(random_state=RS), ExtraTreesClassifier(random_state=RS), GaussianNB(), XGBClassifier(random_state=RS), LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(), LogisticRegression(random_state=RS), MLPClassifier(random_state=RS, activation="logistic"), LGBMClassifier(random_state=RS), CatBoostClassifier(verbose = False)] 


# We will use the stratified K fold Cross Validation to split the data for training because we have an unbalanced datset with minority of target class. 

# In[ ]:


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)


# As first try, we gonna use all different models already set with thier default parameters.
# 
# However, as metric, we will use the recall instead of accuaracy, the reason is we want to detect as much as possible customers from the positive class, i.e customers with hight chance to churn. The accuracy in case of unbalanced data can give us a wrong idea about the target variable, It favors the majority class. 
# 
# Even when this creteria, we must keep a reasonable value of accuracy to avoid classing all cutomers as churners. For this reason, we will choose a threshold that leave a balance between the recall and precision, and we will comapre the results between the default 0.5 threshold and the new one.

# In[ ]:



results = {}
Names=[]
for classifier in classifiers :
    Name = classifier.__class__.__name__
    Names.append(Name)
    AUC=[]
    Recall_Before = []
    Recall_After = []
    Accuaracy_Before = []
    Accuaracy_After = []
    Threshold = []
    Metrics = [AUC , Accuaracy_Before, Accuaracy_After, Threshold,Recall_Before, Recall_After]   
    for train_index, test_index in kfold.split(X.values, Y.values):
        X_train, X_test = X.values[train_index], X.values[test_index]
        Y_train, Y_test = Y.values[train_index], Y.values[test_index]
        classifier.fit(X_train,Y_train)
        FPR, TPR, thresholds = roc_curve(Y_test, classifier.predict_proba(X_test)[:, 1])
        
        Accuaracy_Before.append(accuracy_score(Y_test,classifier.predict(X_test)))
        Recall_Before.append(recall_score(Y_test,classifier.predict(X_test)))
        AUC.append(auc(FPR, TPR))
        
        best_thresh_pos = np.argmax(np.abs(FPR - TPR))
        best_thresh = thresholds[best_thresh_pos]
        Threshold.append(best_thresh)
        
        pred = np.where(classifier.predict_proba(X_test)[:,1]>best_thresh,1,0)

        
        Accuaracy_After.append(accuracy_score(Y_test,pred))
        Recall_After.append(recall_score(Y_test,pred))
        
    results[Name] = [np.mean(L) for L in Metrics]

        
        
        
        


# In[ ]:


ClassifiersRes = pd.DataFrame(results)
ClassifiersRes.index = ["AUC" , "Accuaracy_Before", "Accuaracy_After", "Threshold","Recall_Before", "Recall_After"]
ClassifiersRes = ClassifiersRes.T.sort_values(by = ["Recall_After" , "Accuaracy_After"] , ascending = False)

ClassifiersRes


# It's very clear that as we change the threshold to the optimal one, the recall incearses significantly, and the accuracy decreases but not too much. 
# 
# 4 models are the best in term of model recall: Gradient Boosting Classifier, LGBM Classifier, CatBoost Classifier and Multi Layers Perceptron classifier.
# 
# The question now, is it possible to make those recalls better if we use grid search for each classifier then repeat the same procedure like before. Let's Try to answer that.
# 
# P.S : the following scripts take a very long time to finish, so be patient if you want to check the results for yourself. 

# First we set the hyperparamets for each classifier, then we stack the classifiers and parameters in lists for the for loop.

# In[ ]:


LogReg_params= {"C":np.logspace(-1, 1, 10), "penalty": ["l1","l2"], "solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], "max_iter":[1000]} 

NB_params = {'var_smoothing': np.logspace(0,-9, num=100)}
KNN_params= {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(), "weights": ["uniform","distance"], "metric":["euclidean","manhattan"]} 
SVC_params= {"kernel" : ["rbf"], "gamma": [0.001, 0.01, 0.1], "C": [1,10,100,1000]} 
DT_params = {"min_samples_split" : range(10,500,20), "max_depth": range(1,20,2)} 
RF_params = {"max_features": ["log2","Auto","None"], "min_samples_split":[2,3,5], "bootstrap":[True,False], "n_estimators":[100,150], "criterion":["gini","entropy"]} 
GB_params = {"learning_rate" : [ 0.01, 0.1, 0.05], "n_estimators": [500,1000], "max_depth": [3,5], "min_samples_split": [2,5,10]} 


XGB_params ={ 'n_estimators': [100, 200], 'subsample': [ 0.6, 0.8, 1.0], 'max_depth': [1,2,3], 'learning_rate': [0.1,0.2, 0.3]} 

MLPC_params = {"alpha": [0.1, 0.01, 0.001, 0.0001,0.00001], "hidden_layer_sizes": [(100,100,100), (100,100)], "solver" : ["lbfgs","adam","sgd"], "activation": ["relu","logistic"]} 
CATB_params =  {'depth':[4], 'loss_function': ['CrossEntropy'], 'l2_leaf_reg':np.arange(2,28)} 

LGBM_params = { 'boosting_type':['gbdt','dart','goss','rf'], 'learning_rate':[0.1, 0.2, 0.5, 0.01], 'n_estimators':[100, 500, 1000], 'objective':["binary"]} 

LDA_params = {'solver':['svd', 'lsqr']}

QDA_params = {'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5]}

AdaB_params ={"learning_rate" : [0.001, 0.01, 0.1, 0.05], "algorithm" : ['SAMME', 'SAMME.R'], 'n_estimators':[100, 500, 1000]} 

RS=121
classifiers2 = [LogisticRegression(random_state=RS),GaussianNB(), KNeighborsClassifier(), SVC(random_state=RS,probability=True),DecisionTreeClassifier(random_state=RS), RandomForestClassifier(random_state=RS), GradientBoostingClassifier(random_state=RS), XGBClassifier(random_state=RS), MLPClassifier(random_state=RS), CatBoostClassifier(random_state=RS,verbose = False), LGBMClassifier(random_state=RS), LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(), AdaBoostClassifier(random_state=RS)] 


classifier_params = [LogReg_params,NB_params,KNN_params,SVC_params,DT_params,RF_params, GB_params, XGB_params,MLPC_params,CATB_params, LGBM_params, LDA_params, QDA_params, AdaB_params] 



# In[ ]:




# Tuning by Cross Validation  
cv_result = {}
best_estimators = []

for classifier,classifier_param in zip(classifiers2,classifier_params):
    name = classifier.__class__.__name__
    cv_result[name] = []
    clf = GridSearchCV(classifier, param_grid=classifier_param, cv =10, scoring = "recall", n_jobs = -1) 
    clf.fit(X,Y)
    cv_result[name].append(clf.best_params_)
    best_estimators.append(clf.best_estimator_)    
    
    AUC=[]
    Recall_Before = []
    Recall_After = []
    Accuaracy_Before = []
    Accuaracy_After = []
    Threshold = []
    Metrics = [AUC , Accuaracy_Before, Accuaracy_After, Threshold,Recall_Before, Recall_After]   
    for train_index, test_index in kfold.split(X.values, Y.values):
        X_train, X_test = X.values[train_index], X.values[test_index]
        Y_train, Y_test = Y.values[train_index], Y.values[test_index]
        classifier = clf.best_estimator_.fit(X_train,Y_train)
        FPR, TPR, thresholds = roc_curve(Y_test, classifier.predict_proba(X_test)[:, 1])
        
        Accuaracy_Before.append(accuracy_score(Y_test,classifier.predict(X_test)))
        Recall_Before.append(recall_score(Y_test,classifier.predict(X_test)))
        AUC.append(auc(FPR, TPR))
        
        best_thresh_pos = np.argmax(np.abs(FPR - TPR))
        best_thresh = thresholds[best_thresh_pos]
        Threshold.append(best_thresh)
        
        pred = np.where(classifier.predict_proba(X_test)[:,1]>best_thresh,1,0)

        
        Accuaracy_After.append(accuracy_score(Y_test,pred))
        Recall_After.append(recall_score(Y_test,pred))
        
    cv_result[name].extend([np.mean(L) for L in Metrics])


# In[ ]:


CV_ClassifiersRes = pd.DataFrame(cv_result)
CV_ClassifiersRes.index = ["Params", "AUC" , "Accuaracy_Before", "Accuaracy_After", "Threshold","Recall_Before", "Recall_After"]
CV_ClassifiersRes = CV_ClassifiersRes.T.sort_values(by = ["Recall_After" , "Accuaracy_After"] , ascending = False)

CV_ClassifiersRes


# In the opposite of what we expect the grid search to do, the recalls as not increasing at all, instead, there are decresed.
# 
# So we can say that the best parameters for a model, using recall as metric, not necessary to be the best after changing the threshold. The threshold in this case make a bigger importance for the model performance than the hyperparameters tunning.

# In[ ]:


classifiers11 = [ SVC(random_state=RS, probability = True), AdaBoostClassifier(random_state=RS), GradientBoostingClassifier(random_state=RS), XGBClassifier(random_state=RS), MLPClassifier(random_state=RS, activation="logistic"), LGBMClassifier(random_state=RS), CatBoostClassifier(verbose = False)] 


# In[ ]:


HardVotingEstimator = EnsembleVoteClassifier(classifiers11 , voting='hard')
SoftVotingEstimator = EnsembleVoteClassifier(classifiers11 , voting='soft')

Voting_Classifiers = [HardVotingEstimator]



# In[ ]:


results__hard_vote = {}
classifier = HardVotingEstimator
Name = classifier.__class__.__name__+" "+classifier.voting
Names_vote.append(Name)
AUC=[]
Recall_Before = []
Recall_After = []
Accuaracy_Before = []
Accuaracy_After = []
Threshold = []
Metrics = [AUC , Accuaracy_Before, Accuaracy_After, Threshold,Recall_Before, Recall_After]   
for train_index, test_index in kfold.split(X.values, Y.values):
    X_train, X_test = X.values[train_index], X.values[test_index]
    Y_train, Y_test = Y.values[train_index], Y.values[test_index]
    classifier.fit(X_train,Y_train)
    FPR, TPR, thresholds = roc_curve(Y_test, classifier.predict_proba(X_test)[:, 1])
        
    Accuaracy_Before.append(accuracy_score(Y_test,classifier.predict(X_test)))
    Recall_Before.append(recall_score(Y_test,classifier.predict(X_test)))
    AUC.append(auc(FPR, TPR))
        
    best_thresh_pos = np.argmax(np.abs(FPR - TPR))
    best_thresh = thresholds[best_thresh_pos]
    Threshold.append(best_thresh)
        
    pred = np.where(classifier.predict_proba(X_test)[:,1]>best_thresh,1,0)

        
    Accuaracy_After.append(accuracy_score(Y_test,pred))
    Recall_After.append(recall_score(Y_test,pred))
        
results__hard_vote[Name] = [np.mean(L) for L in Metrics]

        


# In[ ]:


Hard_Voting_ClassifiersRes = pd.DataFrame(results__hard_vote)
Hard_Voting_ClassifiersRes.index = ["AUC" , "Accuaracy_Before", "Accuaracy_After", "Threshold","Recall_Before", "Recall_After"]
Hard_Voting_ClassifiersRes = Hard_Voting_ClassifiersRes.T.sort_values(by = ["Recall_After" , "Accuaracy_After"] , ascending = False)

Hard_Voting_ClassifiersRes


# In[ ]:


results__soft_vote = {}
classifier = SoftVotingEstimator
Name = classifier.__class__.__name__+" "+classifier.voting
Names_vote.append(Name)
AUC=[]
Recall_Before = []
Recall_After = []
Accuaracy_Before = []
Accuaracy_After = []
Threshold = []
Metrics = [AUC , Accuaracy_Before, Accuaracy_After, Threshold,Recall_Before, Recall_After]   
for train_index, test_index in kfold.split(X.values, Y.values):
    X_train, X_test = X.values[train_index], X.values[test_index]
    Y_train, Y_test = Y.values[train_index], Y.values[test_index]
    classifier.fit(X_train,Y_train)
    FPR, TPR, thresholds = roc_curve(Y_test, classifier.predict_proba(X_test)[:, 1])
        
    Accuaracy_Before.append(accuracy_score(Y_test,classifier.predict(X_test)))
    Recall_Before.append(recall_score(Y_test,classifier.predict(X_test)))
    AUC.append(auc(FPR, TPR))
        
    best_thresh_pos = np.argmax(np.abs(FPR - TPR))
    best_thresh = thresholds[best_thresh_pos]
    Threshold.append(best_thresh)
        
    pred = np.where(classifier.predict_proba(X_test)[:,1]>best_thresh,1,0)

        
    Accuaracy_After.append(accuracy_score(Y_test,pred))
    Recall_After.append(recall_score(Y_test,pred))
        
results__soft_vote[Name] = [np.mean(L) for L in Metrics]

        


# In[ ]:


Soft_Voting_ClassifiersRes = pd.DataFrame(results__soft_vote)
Soft_Voting_ClassifiersRes.index = ["AUC" , "Accuaracy_Before", "Accuaracy_After", "Threshold","Recall_Before", "Recall_After"]
Soft_Voting_ClassifiersRes = Soft_Voting_ClassifiersRes.T.sort_values(by = ["Recall_After" , "Accuaracy_After"] , ascending = False)

Soft_Voting_ClassifiersRes


# # Conclusion:
# 
# The model that fit the best our data and make us predict customer churn with the highest recall is Ensemble Vote Classifier with a threshold of 0.234271 and recall of 0.78.
