#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Prediction Analysis

# **Context:** The problem is to look at a churn prediction model for a Telecom customer. We started with understanding of the data presented, followed by a detailed exploratory analysis to look at the different data attributes, their domains, relationships etc. Feature engineering helps in identifying the key variables/set of variables in the context of churners, and help in identifying the ones that impact churning. A baseline model is arrived at based on a set of customer data which is trained further using additional data inputs. Finally the model is tested against the remaining set of data set aside for testing and validation. The resultant validated model is good to go for implementation on actual on-going data sets. 

# ## Exploratory Data Analysis
# 
# Diagrams and graphs are the best way to explore the data, the following plots provide some insghts about the features, their relation with each others, and with our target variable, "Churn".

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from IPython.display import IFrame
print("Telco Customer Churn Analysis")


#  ### Dependent Variable: Churn
# The following is the Churn distribution along all records.

# In[ ]:


IFrame("https://public.tableau.com/views/ChurnesvsNon-Churners/Dashboard2?:embed=y&:showVizHome=no", width=825, height=440)


#  ###  Demographic Features Analysis

# #### Gender
# The dataset contain around 7043 records distributed between females and males. The following diagrame shows the ratios for both in referrence to the "Churn".Its clarify that churn almost equaly distributed between both males and females. Therefor, it's considered has insignificint effect to the target variables.

# In[ ]:


IFrame("https://public.tableau.com/views/ChurnAnalysisbyGender/Dashboard2?:embed=y&:showVizHome=no", width=825, height=470)


# #### Senior Citizen
# Senior citizen feature considered an important variable to predict the churn. The following plots shows that the senior people who churns are double than who dont.

# In[ ]:


IFrame("https://public.tableau.com/views/ChurnAnalysisbySeniorCitizen/Citizen-D?embed=y&:showVizHome=no", width=825, height=440)


# #### Dependents
# Based on the below plot, it looks like that people dont churn if they have dependents. Hence, the dependents feature might have a significient contribution in predicting the "Churn"

# In[ ]:


IFrame("https://public.tableau.com/views/ChurnAnalysisbyDependents/DependentsD?embed=y&:showVizHome=no", width=825, height=455)


# #### Technical Support & Multilines
# Technical support and Multiple lines along the churner's are all explained in the below plot. We can interpretthat the customers with technical support sevrices most probably wont churn. While those who have multiple lines are not willing to churn.

# In[ ]:


IFrame("https://public.tableau.com/views/ChurnbyTechSupportVsMultilines/Dashboard4?:embed=y&:showVizHome=no", width=825, height=800)


# #### Total Charges & Tenures
# When we look at churner data, total charges tend to increase with tenure and this could have been a factor for churning. When we look at the same for non-churners, there doesn’t seem to be a clear increasing trend. 

# In[ ]:


IFrame("https://public.tableau.com/views/TenureVsTotalCharges/Dashboard1?:embed=y&:showVizHome=no", width=825, height=775)


# #### Payment Methode
# People opting for automated payment methods are less likely to churn

# In[ ]:


IFrame("https://public.tableau.com/views/ChurnAnalysisbyPaymentMethodv1/Dashboard1?:embed=y&:showVizHome=no", width=800, height=775)


# ### Gender & Monthly Charges
# In both the Genders, people with high monthly charge tend to churn more than the others. Therefore, we can say that the Monthly Charges feature has a signficient effect to the Churn.

# In[ ]:


IFrame("https://public.tableau.com/views/ChurnbyMonthlychargesVsTenure/Dashboard3?:embed=y&:showVizHome=no", width=800, height=800)


# ###  Contract & Payment Method
# Both One year and Two Year contract tend to keep the churn low, while its the opposit once it comes to the Monthly contract. However, automated payment methods help to keep customers intact even on a month-to-month contract option.

# In[ ]:


IFrame("https://public.tableau.com/views/ChurnAnalysisbyPaymentMethods/Dashboard2?:embed=y&:showVizHome=no", width=800, height=600)


# ### Churn Analysis- Correlation Matrix
# Below chart helps in identifying the right features/variables that are correlated for model building 

# In[ ]:


IFrame("https://public.tableau.com/views/ChurnAnalysis-CorrelationMatrix/ChurnAnalysis-CorrelationMatrix_1?:embed=y&:showVizHome=no", width=800, height=500)


# ### Tenure & Internet Service
# Average duration of non-churners show that if customers stay beyond minimum period they tend to stay longer

# In[ ]:


IFrame("https://public.tableau.com/views/ChurnAnalysisbyServiceProvided/Dashboard3?:embed=y&:showVizHome=no", width=800, height=800)


# ## Data Manipulation
# Data manipuation done on the dataset covers checking the missing values, outliers and NA's. However, in our data set, we found that there are 11 NA's in the "Total Charges" features. Hence, its decided to fill them with zero's since they are considered new and their monthly invoice not issyed yet.

# In[ ]:


IFrame("https://public.tableau.com/views/TotalCharges-Null/TotalCharges-Null?:embed=y&:display_count=yes : embed=y&:showVizHome=no", width=825, height=385)


# In[ ]:


df = pd.read_csv("../../../input/blastchar_telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.columns


# In[ ]:


df['TotalCharges'].fillna(0, inplace=True)
df.isnull().sum()


# ## 2. Baseline
# ### Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ### Importing Dataset

# In[ ]:


df = pd.read_csv("../../../input/blastchar_telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.columns


# In[ ]:


print(df.shape)


# In[ ]:


df.head()


# In[ ]:


df.info()


# Total charges is categorical, it should be changed to float.

# In[ ]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


# In[ ]:


df['TotalCharges'].fillna(0, inplace=True)
df.isnull().sum()


# ### Transform dependent variable to binaries.

# In[ ]:


lb_make = LabelEncoder()
df['Churn_Bi'] = lb_make.fit_transform(df["Churn"])
df[["Churn", "Churn_Bi"]].head()


# ### Remove Churn and Customer ID from the dataset

# In[ ]:


dataset=df.drop('customerID', axis=1)
dataset=dataset.drop('Churn', axis=1)
print(dataset.shape)
dataset.head()


# ### Normalize Continuous variable

# In[ ]:


colToNorm = ('TotalCharges','MonthlyCharges','tenure')
subsetToNormalize = dataset[list(colToNorm)]
print(subsetToNormalize.shape)


# In[ ]:


subsetToKeep = dataset.drop(list(colToNorm), axis=1)


# In[ ]:


print(subsetToKeep.shape)


# In[ ]:


preObj = preprocessing.StandardScaler().fit(subsetToNormalize)
preObj


# In[ ]:


preObj.mean_
preObj.scale_


# In[ ]:


preObj = preObj.transform(subsetToNormalize)


# In[ ]:


preObj = pd.DataFrame(preObj, columns=list(('TotalCharges','MonthlyCharges','tenure')))
subsetToKeep = pd.DataFrame(subsetToKeep)


# In[ ]:


dataset = pd.concat([preObj.reset_index(drop=True), subsetToKeep.reset_index(drop=True)], axis=1)


# ### Creating Dummy Variables 

# In[ ]:


cat_vars=['gender', 'SeniorCitizen','Partner', 'Dependents', 'PhoneService', 'MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract','PaperlessBilling', 'PaymentMethod']


# In[ ]:


for var in cat_vars:
    cat_list='var' + '_' + var
    cat_list=pd.get_dummies(dataset[var],prefix=var)
    dataset_dum=dataset.join(cat_list)
    dataset=dataset_dum


# In[ ]:


dataset.columns


# In[ ]:


dataset_vars=dataset.columns.values.tolist()
to_keep=[i for i in dataset_vars if i not in cat_vars]


# In[ ]:


dataset_v2=dataset[to_keep]
dataset_v2.columns.values


# In[ ]:


dataset_v2.shape


# ### Split, train and Test

# In[ ]:


X = dataset_v2.drop(['Churn_Bi'], axis=1)
y = dataset_v2[['Churn_Bi']]
dataset_v2.shape


# In[ ]:


X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=3457)


# In[ ]:


classifier = LogisticRegression()
classifier.fit(X_train, y_train.values.ravel())


# In[ ]:


y_pred = classifier.predict(X_test)
cf_mx = confusion_matrix(y_test, y_pred)
print(cf_mx)


# In[ ]:


print('Score of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))


# In[ ]:


print(classification_report(y_test, y_pred))


# ## 3. Feature Engineering

# ### Importing Packages

# In[ ]:


import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold # import KFold
#from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.model_selection import cross_validate

import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# ### Importing Dataset

# In[ ]:


df = pd.read_csv("../../../input/blastchar_telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.columns


# In[ ]:


df.head()


# ### Re-arranging features values
# The following features are categorical where each one contains 3 values where 2 of them can be merged into one value. Foe example, in MultipleLines feature, it has values "yes", "No" and "No phone Service's". Therefore, the last two can be merged together as "No".

# In[ ]:


df['MultipleLines']=df['MultipleLines'].replace(['No phone service'], 'No')
df['OnlineSecurity']=df['OnlineSecurity'].replace(['No internet service'], 'No')
df['OnlineBackup']=df['OnlineBackup'].replace(['No internet service'], 'No')
df['DeviceProtection']=df['DeviceProtection'].replace(['No internet service'], 'No')
df['TechSupport']=df['TechSupport'].replace(['No internet service'], 'No')
df['StreamingTV']=df['StreamingTV'].replace(['No internet service'], 'No')
df['StreamingMovies']=df['StreamingMovies'].replace(['No internet service'], 'No')


# ### Converting Fields to Binaries
# Payment Method, Contract and Internet Service are a features with categorical values that have no notion or sense of order. Therefore, we need to transform them into a more representative numerical form at which it can be easily understood by downstream code and pipeline.

# In[ ]:


df['PaymentMethod_echeck'] = np.where(df['PaymentMethod'] == "Electronic check",1,0)
df['PaymentMethod_Mailed'] = np.where(df['PaymentMethod'] == "Mailed check",1,0)
df['PaymentMethod_Transfer'] = np.where(df['PaymentMethod'] == "Bank transfer (automatic)",1,0)
df['PaymentMethod_Crdt'] = np.where(df['PaymentMethod'] == "Credit card (automatic)",1,0)


# In[ ]:


df['MonthlyContract'] = np.where(df['Contract'] == "Month-to-month",1,0)
df['OneYearContract'] = np.where(df['Contract'] == "One year",1,0)
df['TwoYearContract'] = np.where(df['Contract'] == "Two year",1,0)


# In[ ]:


df['InternetService_No'] = np.where(df['InternetService'] == "No",0,0)
df['InternetService_Fibre'] = np.where(df['InternetService'] == "Fiber optic",1,0)
df['InternetService_DSL'] = np.where(df['InternetService'] == "DSL",1,0)


# Converting the other categorical variables to binaries. This is needed to make sure that they will be easily understoor and transformed by the different model algorithems

# In[ ]:


lb_make = LabelEncoder()
df['Churn'] = lb_make.fit_transform(df["Churn"])
df['MultipleLines'] = lb_make.fit_transform(df["MultipleLines"])
df['OnlineSecurity'] = lb_make.fit_transform(df["OnlineSecurity"])
df['gender'] = lb_make.fit_transform(df["gender"])
df['SeniorCitizen'] = lb_make.fit_transform(df["SeniorCitizen"])
df['Partner'] = lb_make.fit_transform(df["Partner"])
df['Dependents'] = lb_make.fit_transform(df["Dependents"])
df['PhoneService'] = lb_make.fit_transform(df["PhoneService"])
df['OnlineBackup'] = lb_make.fit_transform(df["OnlineBackup"])
df['DeviceProtection'] = lb_make.fit_transform(df["DeviceProtection"])
df['TechSupport'] = lb_make.fit_transform(df["TechSupport"])
df['StreamingTV'] = lb_make.fit_transform(df["StreamingTV"])
df['StreamingMovies'] = lb_make.fit_transform(df["StreamingMovies"])
df['PaperlessBilling'] = lb_make.fit_transform(df["PaperlessBilling"])


# Dropping old features, InternetService, Contract and PaymentMethods

# In[ ]:


dataset0=df.drop('InternetService', axis=1)
dataset0 = dataset0.drop('Contract',axis=1)
dataset0 = dataset0.drop('PaymentMethod',axis=1)
dataset0 = dataset0.drop('customerID',axis=1)


# Converting total charges to float, and replacing NA's with zero's

# In[ ]:


dataset0['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
dataset0['TotalCharges'].fillna(0, inplace=True)


# In[ ]:


dataset0.columns


# In[ ]:


dataset0.shape


# In[ ]:


dataset0.info()


# ### Balancing Dataset
# Looking for the dataset balance in term of the Churner's, we can find that the data is not balanced, as the churnners around 25% while the rest dont. Therefore, will do a bootstrap through SMOTE function in Imblearn package.

# In[ ]:


from imblearn.over_sampling import SMOTE
X = dataset0.loc[:, dataset0.columns != 'Churn']
y = dataset0.loc[:, dataset0.columns == 'Churn']
y=pd.DataFrame(y)
os = SMOTE(random_state=0)
columns = X.columns
os_data_X,os_data_y=os.fit_sample(X, y.values.ravel())
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['Churn'])


# In[ ]:


print("length of oversampled data is ",len(os_data_X))
print("Number of No Churn in oversampled data",len(os_data_y[os_data_y['Churn']==0]))
print("Number of Churn",len(os_data_y[os_data_y['Churn']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['Churn']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['Churn']==1])/len(os_data_X))


# In[ ]:


os_data_Xy = pd.concat([os_data_X.reset_index(drop=True), os_data_y.reset_index(drop=True)], axis=1)


# ### Features Construction
# Creating new feature based on a clustering between certain features might add a value toward the final model accuracy, where the target is to find a kind of grouping that split the records according to their churn status. Hence, we will use the K-Mode which is targeting categorical kind of features.

# In[ ]:


os_data_Xy.columns


# In[ ]:


os_data_Xy.shape


# In[ ]:


colToCluster = ('InternetService_Fibre','MonthlyContract','OneYearContract', 'TwoYearContract', 'Churn')


# In[ ]:


subset4Cluster = os_data_Xy[list(colToCluster)]


# In[ ]:


from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array


# In[ ]:


from kmodes import kmodes
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes


# In[ ]:


km = kmodes.KModes(n_clusters=6, init='Cao', n_init=5, verbose=0)


# In[ ]:


clusters = km.fit_predict(subset4Cluster)


# In[ ]:


subset4Cluster=pd.DataFrame(subset4Cluster)


# In[ ]:


subset4Cluster['clusters'] = clusters


# In[ ]:


ct = pd.crosstab(subset4Cluster['clusters'], subset4Cluster['Churn'])


# In[ ]:


print(ct)


# From the above table, we can find that groups 3,4 and 5 sharply split the data in referrence to the Churn. 

# ### Features Selection
# The dataset includes both categorical and nuemerical variables, the following code from sklearn package will classify the features according to their importance, "True" will be considered as an important feature, while the opposite is the "False".

# In[ ]:


os_data_Xy['clusters']=clusters


# In[ ]:


def feature_sel(dataset,model,nb_feat):
    model = LogisticRegression()
    rfe = RFE(model, nb_feat)
    #os_data_y= pd.DataFrame(data=os_data_y,columns=['Churn'])
    os_data_y=dataset.loc[:, dataset.columns == 'Churn']
    os_data_X=dataset.loc[:, dataset.columns != 'Churn']
    rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
    final_vars=os_data_X.columns.values.tolist()
    j=0
    final_list=[]
    for i in final_vars:
        if rfe.support_[j] == True :
            #print(list_2[j])
            final_list.append(i)    
        j = j + 1
    return final_list


# Build a function for Classifications model

# In[ ]:


def run_model(X_f,y_f,model, alg_name, plot_index):
    #X_f = dataset_f.loc[:, dataset_f.columns != 'Churn']
    #y_f = dataset_f.loc[:, dataset_f.columns == 'Churn']
    X_train_f, X_test_f, y_train_f,y_test_f = train_test_split(X_f,y_f,test_size=0.2,random_state=3457)
    model.fit(X_train_f, y_train_f.values.ravel())
    
    if alg_name == "Logistic Regression":
        THRESHOLD = 0.4
        y_pred_f = np.where(model.predict_proba(X_test_f)[:,1] > THRESHOLD, 1, 0)
    else :
        y_pred_f = model.predict(X_test_f)
    #print('Score of ' + alg_name + ' on test set: {:.4f}'.format(model.score(X_test_f, y_test_f)))
    score_f= model.score(X_test_f, y_test_f)
    cf_mx1_f = confusion_matrix(y_test_f, y_pred_f)
    #print(cf_mx1_f)
    True_Pred_f = cf_mx1_f[0][0] + cf_mx1_f[1][1]
    Total_Pred_f = cf_mx1_f[0][0] + cf_mx1_f[1][1] + cf_mx1_f[0][1] + cf_mx1_f[1][0]
    acc= True_Pred_f/Total_Pred_f
    #print('Accuracy of ' + alg_name + ' on test set: {:.4f}'.format(acc))
    return score_f,acc


# **Backup Data Set**

# In[ ]:


#from copy import deepcopy
#os_data_X1 = deepcopy(os_data_X)
#os_data_y1 = deepcopy(os_data_y)


# In[ ]:


from sklearn.feature_selection import RFE


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ### Select number of features based on accuracy

# In[ ]:


alg_name = "Logistic Regression"
model = LogisticRegression(random_state=3457)
score_l=[]
acc_l=[]
for i in range(3,10):
    final_list=feature_sel(os_data_Xy,model,i)
    os_data_y=os_data_Xy.loc[:, os_data_Xy.columns == 'Churn']
    os_data_X=os_data_Xy[final_list]
    #final_list
    score_f,acc_f = run_model(os_data_X,os_data_y,model, "Logistic Regression", 2)
    score_l.append(score_f)
    acc_l.append(acc_f)
    ind_score= score_l.index(max(score_l))
    
print('Number of feature:' + str(ind_score+3))
print('Score of ' + alg_name + ' on test set: {:.4f}'.format(score_l[ind_score]))
print('Accuracy of ' + alg_name + ' on test set: {:.4f}'.format(acc_l[ind_score]))
final_list=feature_sel(os_data_Xy,model,ind_score+3)
final_list
os_data_lg_X=os_data_X[final_list]


# # Logistic Regression with a L1 penalty

# ### Select number of features based on accuracy

# In[ ]:


model = LogisticRegression(random_state=3457,C=0.01, penalty='l1', tol=0.01, solver='liblinear')
alg_name = "Logistic Regression"
#model_lg = LogisticRegression(random_state=3457)
score_l=[]
acc_l=[]
for i in range(3,10):
    final_list=feature_sel(os_data_Xy,model,i)
    os_data_y=os_data_Xy.loc[:, os_data_Xy.columns == 'Churn']
    os_data_X=os_data_Xy[final_list]
    #final_list
    score_f,acc_f = run_model(os_data_X,os_data_y,model, "Logistic Regression", 2)
    score_l.append(score_f)
    acc_l.append(acc_f)
    ind_score= score_l.index(max(score_l))
    
print('Number of feature:' + str(ind_score+3))
print('Score of ' + alg_name + ' on test set: {:.4f}'.format(score_l[ind_score]))
print('Accuracy of ' + alg_name + ' on test set: {:.4f}'.format(acc_l[ind_score]))
final_list=feature_sel(os_data_Xy,model,ind_score+3)
final_list
os_data_lgL1_X=os_data_X[final_list]


# # Random Forest

# ### Select number of features based on accuracy

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from numpy.core.umath_tests import inner1d


# In[ ]:


alg_name = "Random Forest"
model = RandomForestClassifier(random_state=3457)
score_l=[]
acc_l=[]
for i in range(3,10):
    final_list=feature_sel(os_data_Xy,model,i)
    os_data_y=os_data_Xy.loc[:, os_data_Xy.columns == 'Churn']
    os_data_X=os_data_Xy[final_list]
    #final_list
    score_f,acc_f = run_model(os_data_X,os_data_y,model, alg_name, 2)
    score_l.append(score_f)
    acc_l.append(acc_f)
    ind_score= score_l.index(max(score_l))
    
print('Number of feature:' + str(ind_score+3))
print('Score of ' + alg_name + ' on test set: {:.4f}'.format(score_l[ind_score]))
print('Accuracy of ' + alg_name + ' on test set: {:.4f}'.format(acc_l[ind_score]))
final_list=feature_sel(os_data_Xy,model,ind_score+3)
final_list
os_data_rf_X=os_data_X[final_list]


# ### Print Features importance

# In[ ]:


final_list=feature_sel(os_data_Xy,model,9)
os_data_y=os_data_Xy.loc[:, os_data_Xy.columns == 'Churn']
os_data_X=os_data_Xy[final_list]
#final_list
score_f,acc_f = run_model(os_data_X,os_data_y,model, alg_name, 2)
print(model.feature_importances_)


# # Decision Tree

# ### Select number of features based on accuracy

# In[ ]:


from sklearn import tree


# In[ ]:


alg_name = "Decision Tree"
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3,random_state=3458)
score_l=[]
acc_l=[]
for i in range(3,10):
    final_list=feature_sel(os_data_Xy,model,i)
    os_data_y=os_data_Xy.loc[:, os_data_Xy.columns == 'Churn']
    os_data_X=os_data_Xy[final_list]
    #final_list
    score_f,acc_f = run_model(os_data_X,os_data_y,model, alg_name, 2)
    score_l.append(score_f)
    acc_l.append(acc_f)
    ind_score= score_l.index(max(score_l))
    
print('Number of feature:' + str(ind_score+3))
print('Score of ' + alg_name + ' on test set: {:.4f}'.format(score_l[ind_score]))
print('Accuracy of ' + alg_name + ' on test set: {:.4f}'.format(acc_l[ind_score]))
final_list=feature_sel(os_data_Xy,model,ind_score+3)
final_list
os_data_dt_X=os_data_X[final_list]


# ### Print Features importance

# In[ ]:


final_list=feature_sel(os_data_Xy,model,3)
os_data_y=os_data_Xy.loc[:, os_data_Xy.columns == 'Churn']
os_data_X=os_data_Xy[final_list]
#final_list
score_f,acc_f = run_model(os_data_X,os_data_y,model, alg_name, 2)
print(model.feature_importances_)


# In[ ]:


X_train, X_test, y_train,y_test = train_test_split(os_data_dt_X,os_data_y,test_size=0.2,random_state=3457)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=5,random_state=3457,max_depth=4)
model.fit(X_train, y_train.values.ravel())

estimator = model.estimators_[4]
y_pred = model.predict(X_test)
print('Score of  on test set: {:.4f}'.format(model.score(X_test, y_test)))
cf_mx1_f = confusion_matrix(y_test, y_pred)
print(cf_mx1_f)
True_Pred_f = cf_mx1_f[0][0] + cf_mx1_f[1][1]
Total_Pred_f = cf_mx1_f[0][0] + cf_mx1_f[1][1] + cf_mx1_f[0][1] + cf_mx1_f[1][0]
acc= True_Pred_f/Total_Pred_f
print('Accuracy of on test set: {:.4f}'.format(acc))


# In[ ]:


from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot',feature_names = os_data_dt_X.columns,class_names = ['Churn','No Churn'],rounded = True, proportion = False,precision = 2, filled = True)



# In[ ]:


# Convert to png
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=1200'])


# ### Visualize the tree

# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('tree.png'))
plt.axis('off')
print()


# # xgboost

# ### Optimie number of features

# In[ ]:


from xgboost import XGBClassifier

#model_xg = XGBClassifier(random_state=3457)
#run_model(os_data_X,os_data_y,model_xg, "XGBoost", 4)


# In[ ]:


alg_name = "XGBoost"
model = XGBClassifier(random_state=3457)
score_l=[]
acc_l=[]
for i in range(3,10):
    final_list=feature_sel(os_data_Xy,model,i)
    os_data_y=os_data_Xy.loc[:, os_data_Xy.columns == 'Churn']
    os_data_X=os_data_Xy[final_list]
    #final_list
    score_f,acc_f = run_model(os_data_X,os_data_y,model, alg_name, 2)
    score_l.append(score_f)
    acc_l.append(acc_f)
    ind_score= score_l.index(max(score_l))
    
print('Number of feature:' + str(ind_score+3))
print('Score of ' + alg_name + ' on test set: {:.4f}'.format(score_l[ind_score]))
print('Accuracy of ' + alg_name + ' on test set: {:.4f}'.format(acc_l[ind_score]))
final_list=feature_sel(os_data_Xy,model,ind_score+3)
final_list
os_data_xg_X=os_data_X[final_list]


# # SVC

# ### Optimize number of features

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


alg_name = "SVC"
model = SVC(random_state=3458)
score_l=[]
acc_l=[]
for i in range(3,10):
    final_list=feature_sel(os_data_Xy,model,i)
    os_data_y=os_data_Xy.loc[:, os_data_Xy.columns == 'Churn']
    os_data_X=os_data_Xy[final_list]
    #final_list
    score_f,acc_f = run_model(os_data_X,os_data_y,model, alg_name, 2)
    score_l.append(score_f)
    acc_l.append(acc_f)
    ind_score= score_l.index(max(score_l))
    
print('Number of feature:' + str(ind_score+3))
print('Score of ' + alg_name + ' on test set: {:.4f}'.format(score_l[ind_score]))
print('Accuracy of ' + alg_name + ' on test set: {:.4f}'.format(acc_l[ind_score]))
final_list=feature_sel(os_data_Xy,model,ind_score+3)
final_list
os_data_svc_X=os_data_X[final_list]


# # KNN

# ### Optimize number of Features

# In[ ]:


from sklearn import neighbors


# In[ ]:


alg_name = "KNN"
model = neighbors.KNeighborsClassifier()
score_l=[]
acc_l=[]
for i in range(3,10):
    final_list=feature_sel(os_data_Xy,model,i)
    os_data_y=os_data_Xy.loc[:, os_data_Xy.columns == 'Churn']
    os_data_X=os_data_Xy[final_list]
    #final_list
    score_f,acc_f = run_model(os_data_X,os_data_y,model, alg_name, 2)
    score_l.append(score_f)
    acc_l.append(acc_f)
    ind_score= score_l.index(max(score_l))
    
print('Number of feature:' + str(ind_score+3))
print('Score of ' + alg_name + ' on test set: {:.4f}'.format(score_l[ind_score]))
print('Accuracy of ' + alg_name + ' on test set: {:.4f}'.format(acc_l[ind_score]))
final_list=feature_sel(os_data_Xy,model,ind_score+3)
final_list
os_data_knn_X=os_data_X[final_list]


# # Cross Validation
# 

# In[ ]:


from scipy import interp
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold # import KFold
#from sklearn.cross_validation import cross_val_score
import matplotlib.patches as patches


# In[ ]:


os_data_y=np.array(os_data_y)
c, r = os_data_y.shape
os_data_y = os_data_y.reshape(c,)


# In[ ]:


from sklearn.model_selection import cross_val_predict


# #### 1- Cross Validation for Random Forest

# In[ ]:


os_data_y=os_data_Xy.loc[:, os_data_Xy.columns == 'Churn']


# In[ ]:


#from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.model_selection  import cross_val_score, cross_val_predict


# In[ ]:


os_data_y=np.array(os_data_y)
c, r = os_data_y.shape
os_data_y = os_data_y.reshape(c,)


# In[ ]:


def cross_val_f(model,X,y):
    scores_f = cross_val_score(model,X, y, cv=3)
    #print ("Cross-validated scores for Random Forest:", scores_lg)
    print ("Accuracy: %0.2f (+/- %0.2f)" % (scores_f.mean(), scores_f.std() * 2))


# In[ ]:


model_rf = RandomForestClassifier(n_estimators=6,random_state=3457,max_depth=4,max_leaf_nodes=25)
cross_val_f(model_rf,os_data_rf_X,os_data_y)


# #### 2- Cross Validation for Random Logistic Regression

# In[ ]:


model_lg = LogisticRegression(random_state=3457)
cross_val_f(model_lg,os_data_lg_X,os_data_y)


# #### 3- Cross Validation for KNN 

# In[ ]:


model_knn = neighbors.KNeighborsClassifier()
cross_val_f(model_knn,os_data_knn_X,os_data_y)


# In[ ]:


#### 4- Cross Validation for SVC


# In[ ]:


model_svc = SVC(C=1.0, degree=2,random_state=3458,kernel='rbf')
cross_val_f(model_svc,os_data_svc_X,os_data_y)


# #### 5- Cross Validation for XGBOOST

# In[ ]:


model_xg = XGBClassifier(random_state=3457)
cross_val_f(model_xg,os_data_xg_X,os_data_y)


# #### 6- Cross Validation for Logistic Regression

# In[ ]:



model_dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3,random_state=3458)
cross_val_f(model_dt,os_data_dt_X,os_data_y)


# #### 7- Cross Validation for Logistic Regression with L1

# In[ ]:


model_lgL1 = LogisticRegression(random_state=3457,C=0.1, penalty='l1', tol=0.1, solver='liblinear')
cross_val_f(model_lgL1,os_data_lgL1_X,os_data_y)


# ## Print Confusion Matrix and Classification report for Decision Tree
# 

# In[ ]:


#from sklearn.model_selection import cross_val_predict
y_pred= cross_val_predict(model_dt, os_data_dt_X, os_data_y, cv=3)
conf_mat = confusion_matrix(os_data_y, y_pred)
print(conf_mat)


# In[ ]:


print(classification_report(os_data_y, y_pred))


# 

# ## Plot ROC
# 

# In[ ]:


from sklearn.model_selection import StratifiedKFold
#clf = RandomForestClassifier(random_state=3457)
model_dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3,random_state=3458)
cv = StratifiedKFold(n_splits=5,shuffle=False)


# In[ ]:


X_train, X_test, y_train,y_test = train_test_split(os_data_dt_X,os_data_y,test_size=0.2,random_state=3457)


# In[ ]:


model_dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3,random_state=3458)


# In[ ]:


os_data_dt_X=pd.DataFrame(os_data_dt_X)
os_data_y=pd.DataFrame(os_data_y)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 101)
i = 1
for train,test in cv.split(os_data_dt_X,os_data_y):
    prediction = model_dt.fit(os_data_dt_X.iloc[train],os_data_y.iloc[train]).predict_proba(os_data_dt_X.iloc[test])
    fpr, tpr, t = roc_curve(os_data_y.iloc[test], prediction[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i= i+1
plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.text(0.32,0.7,'More accurate area',fontsize = 12)
plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
print()


# ## 5. Summary of the models

# ####  The following table represnts the summary of all models trained and tested. 
# ####  Hence, the metric proposed is the accuracy, and the best perofoming model founded is the RandomeForest.

# In[ ]:


from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score


# In[ ]:


#gives model report in dataframe
def model_report(model,training_x,testing_x,training_y,testing_y,name) :
    model.fit(training_x,training_y)
    #scores_xg = cross_val_score(model_xg,os_data_X, os_data_y, cv=3)
    predictions  = model.predict(testing_x)
    accuracy     = "{0:.4f}".format(accuracy_score(testing_y,predictions))
    recallscore  = recall_score(testing_y,predictions)
    precision    = precision_score(testing_y,predictions)
    roc_auc      = roc_auc_score(testing_y,predictions)
    f1score      = f1_score(testing_y,predictions) 
    kappa_metric = cohen_kappa_score(testing_y,predictions)

    df = pd.DataFrame({"Model"           : [name],"Accuracy_score"  : [accuracy],"Recall_score"    : [recallscore],"Precision"       : [precision],"f1_score"        : [f1score],"Area_under_curve": [roc_auc],"Kappa_metric"    : [kappa_metric],})
    return df


# In[ ]:


def model_report_cv(model,os_data_X, os_data_y,name) :
    
    y_pred= cross_val_predict(model, os_data_X, os_data_y, cv=3)
    accuracy = "{0:.4f}".format(cross_val_score(model,os_data_X, os_data_y, cv=3).mean())
    recallscore = recall_score(os_data_y, y_pred,average='weighted')
    precision     = precision_score(os_data_y, y_pred,average='weighted')
    roc_auc=roc_auc_score(os_data_y, y_pred,average='weighted')
    f1score=f1_score(os_data_y, y_pred,average='weighted')	 
    kappa_metric=cohen_kappa_score(os_data_y, y_pred)	
    df = pd.DataFrame({"Model"           : [name],"Accuracy_score"  : [accuracy],"Recall_score"    : [recallscore],"Precision"       : [precision],"f1_score"        : [f1score],"Area_under_curve": [roc_auc],"Kappa_metric"    : [kappa_metric],})
    return df


# In[ ]:


os_data_bX=dataset0.loc[:, dataset0.columns != 'Churn']
os_data_by=dataset0.loc[:, dataset0.columns == 'Churn']


# In[ ]:



os_data_y=os_data_Xy.loc[:, os_data_Xy.columns == 'Churn']


# In[ ]:


#Base Model

model_base = LogisticRegression(random_state=3457)
train_X, test_X, train_Y,test_Y = train_test_split(os_data_bX,os_data_by.values.ravel(),test_size=0.2,random_state=3457)
model1 = model_report(model_base,train_X,test_X,train_Y,test_Y,"Logistic Regression(BM)")
#model1 = model_report_cv(model_base,os_data_X,os_data_y,"Logistic Regression(BM)")


# In[ ]:


# Linear Regreassion

model_lg = LogisticRegression(random_state=3457)
train_X, test_X, train_Y,test_Y = train_test_split(os_data_lg_X,os_data_y.values.ravel(),test_size=0.2,random_state=3457)
model2 = model_report(model_lg,train_X,test_X,train_Y,test_Y,"Logistic Regression")
#model2 = model_report_cv(model_lg,os_data_lg_X,os_data_y.values.ravel(),"Logistic Regression")


# In[ ]:


model_lgL1 = LogisticRegression(random_state=3457,C=0.1, penalty='l1', tol=0.1, solver='liblinear')
#train_X, test_X, train_Y,test_Y = train_test_split(os_data_lgL1_X,os_data_y,test_size=0.2,random_state=3457)
#model3 = model_report(model_lgL1,train_X,test_X,train_Y,test_Y,"Logistic Regression(L1)")
model3 = model_report_cv(model_lgL1,os_data_lgL1_X,os_data_y.values.ravel(),"Logistic Regression(L1)")


# In[ ]:


#model_dt = RandomForestClassifier(n_estimators=6,random_state=3457,max_depth=4,max_leaf_nodes=25)
model_dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3,random_state=3457)
#model_dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3,random_state=3458,n_estimators=6)
#train_X, test_X, train_Y,test_Y = train_test_split(os_data_dt_X,os_data_y,test_size=0.2,random_state=3457)
#model4 = model_report(model_dt,train_X,test_X,train_Y,test_Y,"Decision Tree")
model4 = model_report_cv(model_dt,os_data_dt_X,os_data_y.values.ravel(),"Decision Tree")


# In[ ]:


model_rf = RandomForestClassifier(n_estimators=6,random_state=3457,max_depth=4,max_leaf_nodes=25)
#train_X, test_X, train_Y,test_Y = train_test_split(os_data_rf_X,os_data_y,test_size=0.2,random_state=3457)
#model5 = model_report(model_rf,train_X,test_X,train_Y,test_Y,"Random Forest")
model5 = model_report_cv(model_rf,os_data_rf_X,os_data_y.values.ravel(),"Random Forest")


# In[ ]:


model_xg = XGBClassifier(random_state=3457)
#train_X, test_X, train_Y,test_Y = train_test_split(os_data_xg_X,os_data_y,test_size=0.2,random_state=3457)
#model6 = model_report(model_xg,train_X,test_X,train_Y,test_Y,"XGBOOST")
model6 = model_report_cv(model_xg,os_data_xg_X,os_data_y.values.ravel(),"XGBOOST")


# In[ ]:


model_svc = SVC(C=0.01, degree=2,random_state=3457,kernel='rbf')
#train_X, test_X, train_Y,test_Y = train_test_split(os_data_svc_X,os_data_y,test_size=0.2,random_state=3457)
#model7 = model_report(model_svc,train_X,test_X,train_Y,test_Y,"SVC")
model7 = model_report_cv(model_svc,os_data_svc_X,os_data_y.values.ravel(),"SVC")


# In[ ]:


model_knn = neighbors.KNeighborsClassifier()
#train_X, test_X, train_Y,test_Y = train_test_split(os_data_knn_X,os_data_y,test_size=0.2,random_state=3457)
#model8 = model_report(model_knn,train_X,test_X,train_Y,test_Y,"KNN")
model8 = model_report_cv(model_knn,os_data_knn_X,os_data_y.values.ravel(),"KNN")


# In[ ]:


import io
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization


# In[ ]:


model_performances = pd.concat([model1,model2,model3,model4,model5,model6,model7,model8],axis = 0).reset_index()

model_performances = model_performances.drop(columns = "index",axis =1)

table  = ff.create_table(np.round(model_performances,4))
py.iplot(table)


# In[ ]:





# 

