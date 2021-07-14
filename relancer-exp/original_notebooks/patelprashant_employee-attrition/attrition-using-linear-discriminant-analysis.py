#!/usr/bin/env python
# coding: utf-8

# # Employee Attrition using Linear Discriminant Analysis

# **Linear Discriminant Analysis (LDA)** is a technique of model distribution of predictors in each of the response classes and use Bayes Theorem to flip around into estimates for classwise probability. This approach assumes the predictors to have come from multivariate Gaussian Distribution with class specific mean vector and common covariance matrix."

# ## Importing Libraries and Dataset

# In[ ]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data = pd.read_csv("../../../input/patelprashant_employee-attrition/WA_Fn-UseC_-HR-Employee-Attrition.csv")
data.head()


# In[ ]:


data.columns


# ### Few of the predictors are useless carrying same value for all the observations ,thus having no significance in the desired output variable:
# ####    'EmployeeCount' , 'EmployeeNumber' , 'Over18' , 'StandardHours

# In[ ]:


data = data.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'],axis=1)
data.columns


# ### Getting unique elements for every predictor variable

# In[ ]:


data['MaritalStatus'].unique()


# ## Replacing our Attrition output by integer constants

# In[ ]:


data.loc[data['Attrition']=='No','Attrition'] = 0
data.loc[data['Attrition']=='Yes','Attrition'] = 1
data.head()


# ## Categorising on the basis of travel for business purposes; rarely,frequently or no travel

# In[ ]:


data['Business_Travel_Rarely']=0
data['Business_Travel_Frequently']=0
data['Business_Non-Travel']=0

data.loc[data['BusinessTravel']=='Travel_Rarely','Business_Travel_Rarely'] = 1
data.loc[data['BusinessTravel']=='Travel_Frequently','Business_Travel_Frequently'] = 1
data.loc[data['BusinessTravel']=='Non-Travel','Business_Non-Travel'] = 1


# ## Categorising on the basis of education field

# In[ ]:


data['Life Sciences']=0
data['Medical']=0
data['Marketing']=0
data['Technical Degree']=0
data['Education Human Resources']=0
data['Education_Other']=0

data.loc[data['EducationField']=='Life Sciences','Life Sciences'] = 1
data.loc[data['EducationField']=='Medical','Medical'] = 1
data.loc[data['EducationField']=='Other','Education_Other'] = 1
data.loc[data['EducationField']=='Technical Degree','Technical Degree'] = 1
data.loc[data['EducationField']=='Human Resources','Education Human Resources'] = 1
data.loc[data['EducationField']=='Marketing','Marketing'] = 1


# ## Categorising on the basis of working department

# In[ ]:


data['Sales']=0
data['R&D']=0
data['Dept_Human Resources'] =0

data.loc[data['Department']=='Sales','Sales'] = 1
data.loc[data['Department']=='Research & Development','R&D'] = 1
data.loc[data['Department']=='Human Resources','Dept_Human Resources'] = 1


# ##  Setting predictor gender where male is indicated as 1 and female as 0

# In[ ]:


data.loc[data['Gender']=='Male','Gender'] = 1
data.loc[data['Gender']=='Female','Gender'] = 0


# ## Categorising on the basis of Job Role

# In[ ]:


data['Research Scientist']=0
data['Laboratory Technician']=0
data['Sales Executive']=0
data['Manufacturing Director']=0
data['Healthcare Representative']=0
data['Sales Representative']=0
data['Research Director']=0
data['Manager'] = 0
data['Job_Human_Resources'] = 0

data.loc[data['JobRole']=='Research Scientist','Research Scientist'] = 1
data.loc[data['JobRole']=='Laboratory Technician','Laboratory Technician'] = 1
data.loc[data['JobRole']=='Sales Executive','Sales Executive'] = 1
data.loc[data['JobRole']=='Sales Representative','Sales Representative'] = 1
data.loc[data['JobRole']=='Manufacturing Director','Manufacturing Director'] = 1
data.loc[data['JobRole']=='Healthcare Representative','Healthcare Representative'] = 1
data.loc[data['JobRole']=='Research Director','Research Director'] = 1
data.loc[data['JobRole']=='Manager','Manager'] = 1
data.loc[data['JobRole']=='Human Resources','Job_Human_Resources'] = 1
data.head()


# ## Categorising on the basis of Marital Satus of Employee

# In[ ]:


data['Marital_single']=0
data['Marital_married']=0
data['Marital_divorced']=0

data.loc[data['MaritalStatus']=='Married','Marital_married'] = 1
data.loc[data['MaritalStatus']=='Single','Marital_single'] = 1
data.loc[data['MaritalStatus']=='Divorced','Marital_divorced'] = 1


# ## Setting up the Over Time predictor

# In[ ]:


data.loc[data['OverTime']=='No','OverTime'] = 0
data.loc[data['OverTime']=='Yes','OverTime'] = 1
data.head()


# ## Checking for useless predictor variables and removing them

# In[ ]:


data.columns


# In[ ]:


data = data.drop(['BusinessTravel','EducationField', 'Department','JobRole','MaritalStatus'],axis=1) 
data.head()


# ## Converting datatypes of some predictor variables

# In[ ]:


data.dtypes


# In[ ]:


data['Attrition'] = data['Attrition'].astype('int')
data['Gender'] = data['Gender'].astype('int')
data['OverTime'] = data['OverTime'].astype('int')


# ## Finding coorelation among various predictors

# In[ ]:


data.corr()


# ## Dividing data into train and test dataset

# In[ ]:


from sklearn.cross_validation import train_test_split
#from random import seed

#seed(20)
train_x = data.drop(['Attrition'],axis=1)
train_y = data['Attrition']

X,test_x,Y,test_y = train_test_split(train_x, train_y, test_size=0.3,random_state=20)
len(test_x)


# ## Applying Linear Discriminant Analysis (LDA) to our data

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()
clf.fit(X,Y)


# ### Calculating accuracy of our model

# In[ ]:


from sklearn.metrics import accuracy_score

pred_y = clf.predict(test_x)

accuracy = accuracy_score(test_y, pred_y, normalize=True, sample_weight=None)
accuracy


# ## Getting quantitative estimates of our model

# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(test_y, pred_y))


# #### We have applied linear discriminant analysis to the data getting an accuracy of 88.20% little bit higher then the logistic approach. As it is clear from our model that we are getting low value of recall for true value of attrition ,i.e., we are not getting enough of the relevant information of the attrited employees from the data. The retrieved model shows high senstivity but low specificity.
# 
# Now ,we will try a new model with few less relevant features trimmed out from our dataset.

# ## Applying Recursive Feature Elimination (RFE) for feature selection

# In[ ]:


from sklearn.feature_selection import RFE

rfe = RFE(clf,40)
rfe = rfe.fit(train_x,train_y)
print(rfe.support_)
print(rfe.ranking_)


# ### Transforming our data to desired no. of features

# In[ ]:


X =rfe.transform(X)
test_x = rfe.transform(test_x)
X.shape


# ### Calculating accuracy of our modified model

# In[ ]:


from sklearn.metrics import accuracy_score

clf.fit(X,Y)
pred_y = clf.predict(test_x)

accuracy = accuracy_score(test_y, pred_y, normalize=True, sample_weight=None)
accuracy


# ### Quantitative estimates of our transformed model

# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(test_y, pred_y))


# #### Thus, we see a slight increase in accuracy of our model. We have trimmed our dataset to 40 features. This approach also shows considerable increase in precision,recall and F1 score .This ultimately results in increase of specificity of our model.This model also reduces our memory space and processing time as the operations to be performed are much less than former.
# 
# Trying with different number of features changes the accuracy of the model.

# ## Applying Quadratic Discriminant Analysis (QDA) to our data

# In[ ]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

clf = QuadraticDiscriminantAnalysis()
clf.fit(X,Y)


# ### Calculating accuracy of our model

# In[ ]:


from sklearn.metrics import accuracy_score

pred_y = clf.predict(test_x)

accuracy = accuracy_score(test_y, pred_y, normalize=True, sample_weight=None)
accuracy


# ### Getting quantitative estimates of our model

# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(test_y, pred_y))


# It is clear that the **Quadratic Discriminant Analysis** is of no use to our data because of smaller dataset and larger variance.Thus, we go with previous two approaches of **Logistic Regression** and **Linear Discriminant Analysis** for our data.

# In[ ]:





# In[ ]:




