#!/usr/bin/env python
# coding: utf-8

# # Employee Churn Model with a Strategic Retention Plan: a HR Analytics Case Study

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Problem-Definition" data-toc-modified-id="Problem-Definition-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Problem Definition</a></span><ul class="toc-item"><li><span><a href="#Project-Overview" data-toc-modified-id="Project-Overview-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Project Overview</a></span></li><li><span><a href="#Problem-Statement" data-toc-modified-id="Problem-Statement-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Problem Statement</a></span></li></ul></li><li><span><a href="#Dataset-Analysis" data-toc-modified-id="Dataset-Analysis-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Dataset Analysis</a></span><ul class="toc-item"><li><span><a href="#Importing-Python-libraries" data-toc-modified-id="Importing-Python-libraries-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Importing Python libraries</a></span></li><li><span><a href="#Importing-the-data" data-toc-modified-id="Importing-the-data-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Importing the data</a></span></li><li><span><a href="#Data-Description-and-Exploratory-Visualisations" data-toc-modified-id="Data-Description-and-Exploratory-Visualisations-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Data Description and Exploratory Visualisations</a></span><ul class="toc-item"><li><span><a href="#Overview" data-toc-modified-id="Overview-2.3.1"><span class="toc-item-num">2.3.1&nbsp;&nbsp;</span>Overview</a></span></li><li><span><a href="#Numerical-features-overview" data-toc-modified-id="Numerical-features-overview-2.3.2"><span class="toc-item-num">2.3.2&nbsp;&nbsp;</span>Numerical features overview</a></span></li></ul></li><li><span><a href="#Feature-distribution-by-target-attribute" data-toc-modified-id="Feature-distribution-by-target-attribute-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Feature distribution by target attribute</a></span><ul class="toc-item"><li><span><a href="#Age" data-toc-modified-id="Age-2.4.1"><span class="toc-item-num">2.4.1&nbsp;&nbsp;</span>Age</a></span></li><li><span><a href="#Education" data-toc-modified-id="Education-2.4.2"><span class="toc-item-num">2.4.2&nbsp;&nbsp;</span>Education</a></span></li><li><span><a href="#Gender" data-toc-modified-id="Gender-2.4.3"><span class="toc-item-num">2.4.3&nbsp;&nbsp;</span>Gender</a></span></li><li><span><a href="#Marital-Status" data-toc-modified-id="Marital-Status-2.4.4"><span class="toc-item-num">2.4.4&nbsp;&nbsp;</span>Marital Status</a></span></li><li><span><a href="#Distance-from-Home" data-toc-modified-id="Distance-from-Home-2.4.5"><span class="toc-item-num">2.4.5&nbsp;&nbsp;</span>Distance from Home</a></span></li><li><span><a href="#Department" data-toc-modified-id="Department-2.4.6"><span class="toc-item-num">2.4.6&nbsp;&nbsp;</span>Department</a></span></li><li><span><a href="#Role-and-Work-Conditions" data-toc-modified-id="Role-and-Work-Conditions-2.4.7"><span class="toc-item-num">2.4.7&nbsp;&nbsp;</span>Role and Work Conditions</a></span></li><li><span><a href="#Years-at-the-Company" data-toc-modified-id="Years-at-the-Company-2.4.8"><span class="toc-item-num">2.4.8&nbsp;&nbsp;</span>Years at the Company</a></span></li><li><span><a href="#Years-With-Current-Manager" data-toc-modified-id="Years-With-Current-Manager-2.4.9"><span class="toc-item-num">2.4.9&nbsp;&nbsp;</span>Years With Current Manager</a></span></li><li><span><a href="#Work-Life-Balance-Score" data-toc-modified-id="Work-Life-Balance-Score-2.4.10"><span class="toc-item-num">2.4.10&nbsp;&nbsp;</span>Work-Life Balance Score</a></span></li><li><span><a href="#Pay/Salary-Employee-Information" data-toc-modified-id="Pay/Salary-Employee-Information-2.4.11"><span class="toc-item-num">2.4.11&nbsp;&nbsp;</span>Pay/Salary Employee Information</a></span></li><li><span><a href="#Employee-Satisfaction-and-Performance-Information" data-toc-modified-id="Employee-Satisfaction-and-Performance-Information-2.4.12"><span class="toc-item-num">2.4.12&nbsp;&nbsp;</span>Employee Satisfaction and Performance Information</a></span></li></ul></li><li><span><a href="#Target-Variable:-Attrition" data-toc-modified-id="Target-Variable:-Attrition-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Target Variable: Attrition</a></span></li><li><span><a href="#Correlation" data-toc-modified-id="Correlation-2.6"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>Correlation</a></span></li><li><span><a href="#EDA-Concluding-Remarks" data-toc-modified-id="EDA-Concluding-Remarks-2.7"><span class="toc-item-num">2.7&nbsp;&nbsp;</span>EDA Concluding Remarks</a></span></li></ul></li><li><span><a href="#Pre-processing-Pipeline" data-toc-modified-id="Pre-processing-Pipeline-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Pre-processing Pipeline</a></span><ul class="toc-item"><li><span><a href="#Encoding" data-toc-modified-id="Encoding-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Encoding</a></span></li><li><span><a href="#Feature-Scaling" data-toc-modified-id="Feature-Scaling-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Feature Scaling</a></span></li><li><span><a href="#Splitting-data-into-training-and-testing-sets" data-toc-modified-id="Splitting-data-into-training-and-testing-sets-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Splitting data into training and testing sets</a></span></li></ul></li><li><span><a href="#Building-Machine-Learning-Models" data-toc-modified-id="Building-Machine-Learning-Models-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Building Machine Learning Models</a></span><ul class="toc-item"><li><span><a href="#Baseline-Algorithms" data-toc-modified-id="Baseline-Algorithms-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Baseline Algorithms</a></span></li><li><span><a href="#Logistic-Regression" data-toc-modified-id="Logistic-Regression-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Logistic Regression</a></span><ul class="toc-item"><li><span><a href="#Fine-tuning" data-toc-modified-id="Fine-tuning-4.2.1"><span class="toc-item-num">4.2.1&nbsp;&nbsp;</span>Fine-tuning</a></span></li><li><span><a href="#Evaluation" data-toc-modified-id="Evaluation-4.2.2"><span class="toc-item-num">4.2.2&nbsp;&nbsp;</span>Evaluation</a></span></li></ul></li><li><span><a href="#Random-Forest-Classifier" data-toc-modified-id="Random-Forest-Classifier-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Random Forest Classifier</a></span><ul class="toc-item"><li><span><a href="#Fine-tuning" data-toc-modified-id="Fine-tuning-4.3.1"><span class="toc-item-num">4.3.1&nbsp;&nbsp;</span>Fine-tuning</a></span></li><li><span><a href="#Evaluation" data-toc-modified-id="Evaluation-4.3.2"><span class="toc-item-num">4.3.2&nbsp;&nbsp;</span>Evaluation</a></span></li></ul></li><li><span><a href="#ROC-Graphs" data-toc-modified-id="ROC-Graphs-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>ROC Graphs</a></span></li></ul></li><li><span><a href="#Concluding-Remarks" data-toc-modified-id="Concluding-Remarks-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Concluding Remarks</a></span><ul class="toc-item"><li><span><a href="#Risk-Category" data-toc-modified-id="Risk-Category-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Risk Category</a></span></li><li><span><a href="#Strategic-Retention-Plan" data-toc-modified-id="Strategic-Retention-Plan-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Strategic Retention Plan</a></span></li></ul></li></ul></div>

# ## Problem Definition

# ### Project Overview

# Employee turn-over (also known as "employee churn") is a costly problem for companies. The true cost of replacing an employee
# can often be quite large. A study by the [Center for American Progress](https://www.americanprogress.org/wp-content/uploads/2012/11/CostofTurnover.pdf) found that companies typically pay about one-fifth of an employee’s salary to replace that employee, and the cost can significantly increase if executives or highest-paid employees are to be replaced. In other words, the cost of replacing employees for most employers remains significant. This is due to the amount of time spent to interview and find a replacement, sign-on bonuses, and the loss of productivity for several months while the new employee gets accustomed to the new role. <br>

# ### Problem Statement

# Understanding why and when employees are most likely to leave can lead to actions to improve employee retention as well as possibly planning new hiring in advance. I will be usign a step-by-step systematic approach using a method that could be used for a variety of ML problems. This project would fall under what is commonly known as "**HR Anlytics**", "**People Analytics**". <br>

# In this study, we will attempt to solve the following problem statement is: <br>
# > ** What is the likelihood of an active employee leaving the company? <br>
# What are the key indicators of an employee leaving the company? <br>
# What policies or strategies can be adopted based on the results to improve employee retention? **
# 
# Given that we have data on former employees, this is a standard **supervised classification problem** where the label is a binary variable, 0 (active employee), 1 (former employee). In this study, our target variable Y is the probability of an employee leaving the company. <br>

# ![title](https://images.unsplash.com/photo-1523006520266-d3a4a8152803?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1422&q=80)

# ## Dataset Analysis

# In this case study, a HR dataset was sourced from [IBM HR Analytics Employee Attrition & Performance](https://www.ibm.com/communities/analytics/watson-analytics-blog/hr-employee-attrition/) which contains employee data for 1,470 employees with various information about the employees. I will use this dataset to predict when employees are going to quit by understanding the main drivers of employee churn. <br>
# 
# As stated on the [IBM website](https://www.ibm.com/communities/analytics/watson-analytics-blog/hr-employee-attrition/) *"This is a fictional data set created by IBM data scientists"*. Its main purpose was to demonstrate the IBM Watson Analytics tool for employee attrition.

# ### Importing Python libraries

# In[ ]:


# importing libraries for data handling and analysis
import pandas as pd
from pandas.plotting import scatter_matrix
from pandas import ExcelWriter
from pandas import ExcelFile
from openpyxl import load_workbook
import numpy as np
from scipy.stats import norm, skew
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# importing libraries for data visualisations
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib
color = sns.color_palette()
from IPython.display import display
pd.options.display.max_columns = None
# Standard plotly imports
from plotly import plotly as py
import plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
#py.initnotebookmode(connected=True) # this code, allow us to work with offline plotly version
# Using plotly + cufflinks in offline mode
import cufflinks as cf
cf.set_config_file(offline=True)
import cufflinks
cufflinks.go_offline(connected=True)


# In[ ]:


# sklearn modules for preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from imblearn.over_sampling import SMOTE  # SMOTE
# sklearn modules for ML model selection
from sklearn.model_selection import train_test_split  # import 'train_test_split'
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Libraries for data modelling
from sklearn import svm, tree, linear_model, neighbors
from sklearn import naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Common sklearn Model Helpers
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
# from sklearn.datasets import make_classification

# sklearn modules for performance metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve, recall_score, log_loss
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import average_precision_score


# In[ ]:


# importing misceallenous libraries
import os
import re
import sys
import timeit
import string
from datetime import datetime
from time import time
from dateutil.parser import parse
# ip = get_ipython()
# ip.register_magics(jupyternotify.JupyterNotifyMagics)


# ### Importing the data

# > Let's import the dataset and make of a copy of the source file for this analysis. <br> The dataset contains 1,470 rows and 35 columns.

# In[ ]:


import os
print(os.listdir("../../../input/pavansubhasht_ibm-hr-analytics-attrition-dataset"))


# In[ ]:


# Read Excel file
df_sourcefile = pd.read_csv("../../../input/pavansubhasht_ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
print("Shape of dataframe is: {}".format(df_sourcefile.shape))


# In[ ]:


# Make a copy of the original sourcefile
df_HR = df_sourcefile.copy()


# ### Data Description and Exploratory Visualisations

# > In this section, we will provide data visualizations that summarizes or extracts relevant characteristics of features in our dataset. Let's look at each column in detail, get a better understanding of the dataset, and group them together when appropriate.

# #### Overview

# In[ ]:


# Dataset columns
df_HR.columns


# In[ ]:


# Dataset header
df_HR.head()


# > The dataset contains several numerical and categorical columns providing various information on employee's personal and employment details.

# In[ ]:


# let's break down the columns by their type (i.e. int64, float64, object)
df_HR.columns.to_series().groupby(df_HR.dtypes).groups


# In[ ]:


# Columns datatypes and missign values
df_HR.info()


# > The data provided has no missing values. In HR Analytics, employee data is unlikely to feature large ratio of missing values as HR Departments typically have all personal and employment data on-file. However, the type of documentation data is being kept in (i.e. whether it is paper-based, Excel spreadhsheets, databases, etc) has a massive impact on the accuracy and the ease of access to the HR data.

# #### Numerical features overview

# In[ ]:


df_HR.describe()


# In[ ]:


df_HR.hist(figsize=(20,20))
print()


# > A few observations can be made based on the information and histograms for numerical features:
#  - Many histograms are tail-heavy; indeed several distributions are right-skewed (e.g. MonthlyIncome DistanceFromHome, YearsAtCompany). Data transformation methods may be required to approach a normal distribution prior to fitting a model to the data.
#  - Age distribution is a slightly right-skewed normal distribution with the bulk of the staff between 25 and 45 years old.
#  - EmployeeCount and StandardHours are constant values for all employees. They're likely to be redundant features.
#  - Employee Number is likely to be a unique identifier for employees given the feature's quasi-uniform distribution.

# ### Feature distribution by target attribute

# #### Age

# > The age distributions for Active and Ex-employees only differs by one year. <br>
# The average age of ex-employees is **33.6** years old, while **37.6** is the average age for current employees.

# In[ ]:

(mu, sigma) = norm.fit(df_HR.loc[df_HR['Attrition'] == 'Yes', 'Age'])
print('Ex-exmployees: average age = {:.1f} years old and standard deviation = {:.1f}'.format(mu, sigma))
(mu, sigma) = norm.fit(df_HR.loc[df_HR['Attrition'] == 'No', 'Age'])
print('Current exmployees: average age = {:.1f} years old and standard deviation = {:.1f}'.format(mu, sigma))


# > Let's create a kernel density estimation (KDE) plot colored by the value of the target. A kernel density estimation (KDE) is a non-parametric way to estimate the probability density function of a random variable. It will allow us to identify if there is a correlation between the Age of the Client and their ability to pay it back.

# In[ ]:


plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'No', 'Age'], label = 'Active Employee')
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'Yes', 'Age'], label = 'Ex-Employees')
plt.xlim(left=18, right=60)
plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.title('Age Distribution in Percent by Attrition Status')


# #### Education

# > Several Education Fields are represented in the dataset, namely: Human Resources, Life Sciences, Marketing, Medical, Technical Degree, and a miscellaneous category Other. Here, I plot the normalized % of Leavers for each Education Field.

# In[ ]:


# Education Field of employees
df_HR['EducationField'].value_counts()


# In[ ]:


df_EducationField = pd.DataFrame(columns=["Field", "% of Leavers"])
i=0
for field in list(df_HR['EducationField'].unique()):
    ratio = df_HR[(df_HR['EducationField']==field)&(df_HR['Attrition']=="Yes")].shape[0] / df_HR[df_HR['EducationField']==field].shape[0]
    df_EducationField.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_EF = df_EducationField.groupby(by="Field").sum()
df_EF.iplot(kind='bar',title='Leavers by Education Field (%)')


# #### Gender

# > Gender distribution shows that the dataset features a higher relative proportion of male ex-employees than female ex-employees, with normalised gender distribution of ex-employees in the dataset at 17.0% for Males and 14.8% for Females.

# In[ ]:


# Gender of employees
df_HR['Gender'].value_counts()


# In[ ]:


print("Normalised gender distribution of ex-employees in the dataset: Male = {:.1f}%; Female {:.1f}%.".format((df_HR[(df_HR['Attrition'] == 'Yes') &(df_HR['Gender'] == 'Male')].shape[0] / df_HR[df_HR['Gender'] == 'Male'].shape[0])*100, (df_HR[(df_HR['Attrition'] == 'Yes') & (df_HR['Gender'] == 'Female')].shape[0] / df_HR[df_HR['Gender'] == 'Female'].shape[0])*100))


# In[ ]:


df_Gender = pd.DataFrame(columns=["Gender", "% of Leavers"])
i=0
for field in list(df_HR['Gender'].unique()):
    ratio = df_HR[(df_HR['Gender']==field)&(df_HR['Attrition']=="Yes")].shape[0] / df_HR[df_HR['Gender']==field].shape[0]
    df_Gender.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_G = df_Gender.groupby(by="Gender").sum()
df_G.iplot(kind='bar',title='Leavers by Gender (%)')


# #### Marital Status

# > The dataset features three marital status: Married (673 employees), Single (470 employees), Divorced (327 employees). <br>
# Single employees show the largest proportion of leavers at 25%.

# In[ ]:


# Marital Status of employees
df_HR['MaritalStatus'].value_counts()


# In[ ]:


df_Marital = pd.DataFrame(columns=["Marital Status", "% of Leavers"])
i=0
for field in list(df_HR['MaritalStatus'].unique()):
    ratio = df_HR[(df_HR['MaritalStatus']==field)&(df_HR['Attrition']=="Yes")].shape[0] / df_HR[df_HR['MaritalStatus']==field].shape[0]
    df_Marital.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_MF = df_Marital.groupby(by="Marital Status").sum()
df_MF.iplot(kind='bar',title='Leavers by Marital Status (%)')


# #### Distance from Home

# > Distance from home for employees to get to work varies from 1 to 29 miles. There is no discernable strong correlation between Distance from Home and Attrition Status as per the KDE plot below.

# In[ ]:


# Distance from Home
print("Distance from home for employees to get to work is from {} to {} miles.".format(df_HR['DistanceFromHome'].min(),df_HR['DistanceFromHome'].max()))


# In[ ]:


print('Average distance from home for currently active employees: {:.2f} miles and ex-employees: {:.2f} miles'.format(df_HR[df_HR['Attrition'] == 'No']['DistanceFromHome'].mean(), df_HR[df_HR['Attrition'] == 'Yes']['DistanceFromHome'].mean()))


# In[ ]:


plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'No', 'DistanceFromHome'], label = 'Active Employee')
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'Yes', 'DistanceFromHome'], label = 'Ex-Employees')
plt.xlabel('DistanceFromHome')
plt.xlim(left=0)
plt.ylabel('Density')
plt.title('Distance From Home Distribution in Percent by Attrition Status')


# #### Department

# > The data features employee data from three departments: Research & Development, Sales, and Human Resources.

# In[ ]:


# The organisation consists of several departments
df_HR['Department'].value_counts()


# In[ ]:


df_Department = pd.DataFrame(columns=["Department", "% of Leavers"])
i=0
for field in list(df_HR['Department'].unique()):
    ratio = df_HR[(df_HR['Department']==field)&(df_HR['Attrition']=="Yes")].shape[0] / df_HR[df_HR['Department']==field].shape[0]
    df_Department.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_DF = df_Department.groupby(by="Department").sum()
df_DF.iplot(kind='bar',title='Leavers by Department (%)')


# #### Role and Work Conditions

# > A preliminary look at the relationship between Business Travel frequency and Attrition Status shows that there is a largest normalized proportion of Leavers for employees that travel "frequently". Travel metrics associated with Business Travel status were not disclosed (i.e. how many hours of Travel is considered "Frequent").

# In[ ]:


# Employees have different business travel commitmnent depending on their roles and level in the organisation
df_HR['BusinessTravel'].value_counts()


# In[ ]:


df_BusinessTravel = pd.DataFrame(columns=["Business Travel", "% of Leavers"])
i=0
for field in list(df_HR['BusinessTravel'].unique()):
    ratio = df_HR[(df_HR['BusinessTravel']==field)&(df_HR['Attrition']=="Yes")].shape[0] / df_HR[df_HR['BusinessTravel']==field].shape[0]
    df_BusinessTravel.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_BT = df_BusinessTravel.groupby(by="Business Travel").sum()
df_BT.iplot(kind='bar',title='Leavers by Business Travel (%)')


# > Several Job Roles are listed in the dataset: Sales Executive, Research Scientist, Laboratory Technician, Manufacturing Director, Healthcare Representative, Manager, Sales Representative, Research Director, Human Resources.

# In[ ]:


# Employees in the database have several roles on-file
df_HR['JobRole'].value_counts()


# In[ ]:


df_JobRole = pd.DataFrame(columns=["Job Role", "% of Leavers"])
i=0
for field in list(df_HR['JobRole'].unique()):
    ratio = df_HR[(df_HR['JobRole']==field)&(df_HR['Attrition']=="Yes")].shape[0] / df_HR[df_HR['JobRole']==field].shape[0]
    df_JobRole.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_JR = df_JobRole.groupby(by="Job Role").sum()
df_JR.iplot(kind='bar',title='Leavers by Job Role (%)')


# > Employees have an assigned level within the organisation which varies from 1 (staff) to 5 (managerial/director). Employees with an assigned Job Level of "1" show the largest normalized proportion of Leavers.

# In[ ]:


df_HR['JobLevel'].value_counts()


# In[ ]:


df_JobLevel = pd.DataFrame(columns=["Job Level", "% of Leavers"])
i=0
for field in list(df_HR['JobLevel'].unique()):
    ratio = df_HR[(df_HR['JobLevel']==field)&(df_HR['Attrition']=="Yes")].shape[0] / df_HR[df_HR['JobLevel']==field].shape[0]
    df_JobLevel.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_JL = df_JobLevel.groupby(by="Job Level").sum()
df_JL.iplot(kind='bar',title='Leavers by Job Level (%)')


# > A ranking is associated to the employee's Job Involvement :1 'Low' 2 'Medium' 3 'High' 4 'Very High'. The plot below indicates a negative correlation with the Job Involvement of an employee and the Attrition Status. In other words, employees with higher Job Involvement are less likely to leave.

# In[ ]:


df_HR['JobInvolvement'].value_counts()


# In[ ]:


df_JobInvolvement = pd.DataFrame(columns=["Job Involvement", "% of Leavers"])
i=0
for field in list(df_HR['JobInvolvement'].unique()):
    ratio = df_HR[(df_HR['JobInvolvement']==field)&(df_HR['Attrition']=="Yes")].shape[0] / df_HR[df_HR['JobInvolvement']==field].shape[0]
    df_JobInvolvement.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_JI = df_JobInvolvement.groupby(by="Job Involvement").sum()
df_JI.iplot(kind='bar',title='Leavers by Job Involvement (%)')


# > The data indicates that employees may have access to some Training. A feature indicates how many years it's been since the employee attended such training.

# In[ ]:


print("Number of training times last year varies from {} to {} years.".format(df_HR['TrainingTimesLastYear'].min(), df_HR['TrainingTimesLastYear'].max()))


# In[ ]:


plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'No', 'TrainingTimesLastYear'], label = 'Active Employee')
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'Yes', 'TrainingTimesLastYear'], label = 'Ex-Employees')
plt.xlabel('TrainingTimesLastYear')
plt.ylabel('Density')
plt.title('Training Times Last Year Distribution in Percent by Attrition Status')


# > There is a feature for the number of companies the employee has worked at. <br>
# > 0 likely indicates that according to records, the employee has only worked at this company

# In[ ]:


df_HR['NumCompaniesWorked'].value_counts()


# In[ ]:


df_NumCompaniesWorked = pd.DataFrame(columns=["Num Companies Worked", "% of Leavers"])
i=0
for field in list(df_HR['NumCompaniesWorked'].unique()):
    ratio = df_HR[(df_HR['NumCompaniesWorked']==field)&(df_HR['Attrition']=="Yes")].shape[0] / df_HR[df_HR['NumCompaniesWorked']==field].shape[0]
    df_NumCompaniesWorked.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_NC = df_NumCompaniesWorked.groupby(by="Num Companies Worked").sum()
df_NC.iplot(kind='bar',title='Leavers by Num Companies Worked (%)')


# #### Years at the Company

# In[ ]:


print("Number of Years at the company varies from {} to {} years.".format(df_HR['YearsAtCompany'].min(), df_HR['YearsAtCompany'].max()))


# In[ ]:


plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'No', 'YearsAtCompany'], label = 'Active Employee')
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'Yes', 'YearsAtCompany'], label = 'Ex-Employees')
plt.xlabel('YearsAtCompany')
plt.xlim(left=0)
plt.ylabel('Density')
plt.title('Years At Company in Percent by Attrition Status')


# In[ ]:


print("Number of Years in the current role varies from {} to {} years.".format(df_HR['YearsInCurrentRole'].min(), df_HR['YearsInCurrentRole'].max()))


# In[ ]:


plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'No', 'YearsInCurrentRole'], label = 'Active Employee')
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'Yes', 'YearsInCurrentRole'], label = 'Ex-Employees')
plt.xlabel('YearsInCurrentRole')
plt.xlim(left=0)
plt.ylabel('Density')
plt.title('Years In Current Role in Percent by Attrition Status')


# In[ ]:


print("Number of Years since last promotion varies from {} to {} years.".format(df_HR['YearsSinceLastPromotion'].min(), df_HR['YearsSinceLastPromotion'].max()))


# In[ ]:


plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'No', 'YearsSinceLastPromotion'], label = 'Active Employee')
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'Yes', 'YearsSinceLastPromotion'], label = 'Ex-Employees')
plt.xlabel('YearsSinceLastPromotion')
plt.xlim(left=0)
plt.ylabel('Density')
plt.title('Years Since Last Promotion in Percent by Attrition Status')


# In[ ]:


print("Total working years varies from {} to {} years.".format(df_HR['TotalWorkingYears'].min(), df_HR['TotalWorkingYears'].max()))


# In[ ]:


plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'No', 'TotalWorkingYears'], label = 'Active Employee')
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'Yes', 'TotalWorkingYears'], label = 'Ex-Employees')
plt.xlabel('TotalWorkingYears')
plt.xlim(left=0)
plt.ylabel('Density')
plt.title('Total Working Years in Percent by Attrition Status')


# #### Years With Current Manager

# In[ ]:


print("Number of Years wit current manager varies from {} to {} years.".format(df_HR['YearsWithCurrManager'].min(), df_HR['YearsWithCurrManager'].max()))


# In[ ]:


plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'No', 'YearsWithCurrManager'], label = 'Active Employee')
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'Yes', 'YearsWithCurrManager'], label = 'Ex-Employees')
plt.xlabel('YearsWithCurrManager')
plt.xlim(left=0)
plt.ylabel('Density')
plt.title('Years With Curr Manager in Percent by Attrition Status')


# #### Work-Life Balance Score

# > A feature related to "Work-Life Balance" was captured as: 1 'Bad' 2 'Good' 3 'Better' 4 'Best'. The data indicates that the largest normalised proportion of Leavers had "Bad" Work-Life Balance.

# In[ ]:


df_HR['WorkLifeBalance'].value_counts()


# In[ ]:


df_WorkLifeBalance = pd.DataFrame(columns=["WorkLifeBalance", "% of Leavers"])
i=0
for field in list(df_HR['WorkLifeBalance'].unique()):
    ratio = df_HR[(df_HR['WorkLifeBalance']==field)&(df_HR['Attrition']=="Yes")].shape[0] / df_HR[df_HR['WorkLifeBalance']==field].shape[0]
    df_WorkLifeBalance.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_WLB = df_WorkLifeBalance.groupby(by="WorkLifeBalance").sum()
df_WLB.iplot(kind='bar',title='Leavers by WorkLifeBalance (%)')


# > All employees have a standard 80-hour work commitment

# In[ ]:


df_HR['StandardHours'].value_counts()


# > Some employees have overtime commitments. The data clearly show that there is significant larger portion of employees with OT that have left the company.

# In[ ]:


df_HR['OverTime'].value_counts()


# In[ ]:


df_OverTime = pd.DataFrame(columns=["OverTime", "% of Leavers"])
i=0
for field in list(df_HR['OverTime'].unique()):
    ratio = df_HR[(df_HR['OverTime']==field)&(df_HR['Attrition']=="Yes")].shape[0] / df_HR[df_HR['OverTime']==field].shape[0]
    df_OverTime.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_OT = df_OverTime.groupby(by="OverTime").sum()
df_OT.iplot(kind='bar',title='Leavers by OverTime (%)')


# #### Pay/Salary Employee Information

# In[ ]:


print("Employee Hourly Rate varies from ${} to ${}.".format(df_HR['HourlyRate'].min(), df_HR['HourlyRate'].max()))


# In[ ]:


print("Employee Daily Rate varies from ${} to ${}.".format(df_HR['DailyRate'].min(), df_HR['DailyRate'].max()))


# In[ ]:


print("Employee Monthly Rate varies from ${} to ${}.".format(df_HR['MonthlyRate'].min(), df_HR['MonthlyRate'].max()))


# In[ ]:


print("Employee Monthly Income varies from ${} to ${}.".format(df_HR['MonthlyIncome'].min(), df_HR['MonthlyIncome'].max()))


# In[ ]:


plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'No', 'MonthlyIncome'], label = 'Active Employee')
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'Yes', 'MonthlyIncome'], label = 'Ex-Employees')
plt.xlabel('Monthly Income')
plt.xlim(left=0)
plt.ylabel('Density')
plt.title('Monthly Income in Percent by Attrition Status')


# In[ ]:


print("Percentage Salary Hikes varies from {}% to {}%.".format(df_HR['PercentSalaryHike'].min(), df_HR['PercentSalaryHike'].max()))


# In[ ]:


plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'No', 'PercentSalaryHike'], label = 'Active Employee')
sns.kdeplot(df_HR.loc[df_HR['Attrition'] == 'Yes', 'PercentSalaryHike'], label = 'Ex-Employees')
plt.xlabel('PercentSalaryHike')
plt.xlim(left=0)
plt.ylabel('Density')
plt.title('Percent Salary Hike in Percent by Attrition Status')


# In[ ]:


print("Stock Option Levels varies from {} to {}.".format(df_HR['StockOptionLevel'].min(), df_HR['StockOptionLevel'].max()))


# In[ ]:


print("Normalised percentage of leavers by Stock Option Level: 1: {:.2f}%, 2: {:.2f}%, 3: {:.2f}%".format(df_HR[(df_HR['Attrition'] == 'Yes') & (df_HR['StockOptionLevel'] == 1)].shape[0] / df_HR[df_HR['StockOptionLevel'] == 1].shape[0]*100,df_HR[(df_HR['Attrition'] == 'Yes') & (df_HR['StockOptionLevel'] == 2)].shape[0] / df_HR[df_HR['StockOptionLevel'] == 1].shape[0]*100,df_HR[(df_HR['Attrition'] == 'Yes') & (df_HR['StockOptionLevel'] == 3)].shape[0] / df_HR[df_HR['StockOptionLevel'] == 1].shape[0]*100))


# In[ ]:


df_StockOptionLevel = pd.DataFrame(columns=["StockOptionLevel", "% of Leavers"])
i=0
for field in list(df_HR['StockOptionLevel'].unique()):
    ratio = df_HR[(df_HR['StockOptionLevel']==field)&(df_HR['Attrition']=="Yes")].shape[0] / df_HR[df_HR['StockOptionLevel']==field].shape[0]
    df_StockOptionLevel.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_SOL = df_StockOptionLevel.groupby(by="StockOptionLevel").sum()
df_SOL.iplot(kind='bar',title='Leavers by Stock Option Level (%)')


# #### Employee Satisfaction and Performance Information

# > Environment Satisfaction was captured as: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'. <br> 
# Proportion of Leaving Employees decreases as the Environment Satisfaction score increases.

# In[ ]:


df_HR['EnvironmentSatisfaction'].value_counts()


# In[ ]:


df_EnvironmentSatisfaction = pd.DataFrame(columns=["EnvironmentSatisfaction", "% of Leavers"])
i=0
for field in list(df_HR['EnvironmentSatisfaction'].unique()):
    ratio = df_HR[(df_HR['EnvironmentSatisfaction']==field)&(df_HR['Attrition']=="Yes")].shape[0] / df_HR[df_HR['EnvironmentSatisfaction']==field].shape[0]
    df_EnvironmentSatisfaction.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_Env = df_EnvironmentSatisfaction.groupby(by="EnvironmentSatisfaction").sum()
df_Env.iplot(kind='bar',title='Leavers by Environment Satisfaction (%)')


# > Job Satisfaction was captured as: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'. <br> 
# Proportion of Leaving Employees decreases as the Job Satisfaction score increases.

# In[ ]:


# Job Satisfaction was captured as: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
df_HR['JobSatisfaction'].value_counts()


# In[ ]:


df_JobSatisfaction = pd.DataFrame(columns=["JobSatisfaction", "% of Leavers"])
i=0
for field in list(df_HR['JobSatisfaction'].unique()):
    ratio = df_HR[(df_HR['JobSatisfaction']==field)&(df_HR['Attrition']=="Yes")].shape[0] / df_HR[df_HR['JobSatisfaction']==field].shape[0]
    df_JobSatisfaction.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_JS = df_JobSatisfaction.groupby(by="JobSatisfaction").sum()
df_JS.iplot(kind='bar',title='Leavers by Job Satisfaction (%)')


# > Relationship Satisfaction was captured as: 1 'Low', 2 'Medium', 3 'High', 4 'Very High'.

# In[ ]:


df_HR['RelationshipSatisfaction'].value_counts()


# In[ ]:


df_RelationshipSatisfaction = pd.DataFrame(columns=["RelationshipSatisfaction", "% of Leavers"])
i=0
for field in list(df_HR['RelationshipSatisfaction'].unique()):
    ratio = df_HR[(df_HR['RelationshipSatisfaction']==field)&(df_HR['Attrition']=="Yes")].shape[0] / df_HR[df_HR['RelationshipSatisfaction']==field].shape[0]
    df_RelationshipSatisfaction.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_RS = df_RelationshipSatisfaction.groupby(by="RelationshipSatisfaction").sum()
df_RS.iplot(kind='bar',title='Leavers by Relationship Satisfaction (%)')


# > Employee Performance Rating was captured as: 1 'Low' 2 'Good' 3 'Excellent' 4 'Outstanding'

# In[ ]:


df_HR['PerformanceRating'].value_counts()


# In[ ]:


print("Normalised percentage of leavers by Stock Option Level: 3: {:.2f}%, 4: {:.2f}%".format(df_HR[(df_HR['Attrition'] == 'Yes') & (df_HR['PerformanceRating'] == 3)].shape[0] / df_HR[df_HR['StockOptionLevel'] == 1].shape[0]*100,df_HR[(df_HR['Attrition'] == 'Yes') & (df_HR['PerformanceRating'] == 4)].shape[0] / df_HR[df_HR['StockOptionLevel'] == 1].shape[0]*100))


# In[ ]:


df_PerformanceRating = pd.DataFrame(columns=["PerformanceRating", "% of Leavers"])
i=0
for field in list(df_HR['PerformanceRating'].unique()):
    ratio = df_HR[(df_HR['PerformanceRating']==field)&(df_HR['Attrition']=="Yes")].shape[0] / df_HR[df_HR['PerformanceRating']==field].shape[0]
    df_PerformanceRating.loc[i] = (field, ratio*100)
    i += 1
    #print("In {}, the ratio of leavers is {:.2f}%".format(field, ratio*100))    
df_PR = df_PerformanceRating.groupby(by="PerformanceRating").sum()
df_PR.iplot(kind='bar',title='Leavers by Performance Rating (%)')


# ### Target Variable: Attrition

# > The feature 'Attrition' is what this Machine Learning problem is about. We are trying to predict the value of the feature 'Attrition' by using other related features associated with the employee's personal and professional history. 

# In[ ]:


# Attrition indicates if the employee is currently active ('No') or has left the company ('Yes')
df_HR['Attrition'].value_counts()


# In[ ]:


print("Percentage of Current Employees is {:.1f}% and of Ex-employees is: {:.1f}%".format(df_HR[df_HR['Attrition'] == 'No'].shape[0] / df_HR.shape[0]*100,df_HR[df_HR['Attrition'] == 'Yes'].shape[0] / df_HR.shape[0]*100))


# In[ ]:


df_HR['Attrition'].iplot(kind='hist', xTitle='Attrition',yTitle='count', title='Attrition Distribution')


# > As shown on the chart above, we see this is an imbalanced class problem. Indeed, the percentage of Current Employees in our dataset is 83.9% and the percentage of Ex-employees is: 16.1%
# 
# > Machine learning algorithms typically work best when the number of instances of each classes are roughly equal. We will have to address this target feature imbalance prior to implementing our Machine Learning algorithms.

# ### Correlation

# > Let's take a look at some of most significant correlations. It is worth remembering that correlation coefficients only measure linear correlations.

# In[ ]:


# Find correlations with the target and sort
df_HR_trans = df_HR.copy()
df_HR_trans['Target'] = df_HR_trans['Attrition'].apply(lambda x: 0 if x == 'No' else 1)
df_HR_trans = df_HR_trans.drop(['Attrition', 'EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1)
correlations = df_HR_trans.corr()['Target'].sort_values()
print('Most Positive Correlations: \n', correlations.tail(5))
print('\nMost Negative Correlations: \n', correlations.head(5))


# > Let's plot a heatmap to visualize the correlation between Attrition and these factors.

# In[ ]:


# Calculate correlations
corr = df_HR_trans.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
# Heatmap
plt.figure(figsize=(15, 10))


# > As shown above, "Monthly Rate", "Number of Companies Worked" and "Distance From Home" are positively correlated to Attrition; <br> while "Total Working Years", "Job Level", and "Years In Current Role" are negatively correlated to Attrition.

# ### EDA Concluding Remarks

# Let's summarise the findings from this EDA: <br>
# 
# > - The dataset does not feature any missing or erroneous data values, and all features are of the correct data type. <br>
# - The strongest positive correlations with the target features are: **Performance Rating**, **Monthly Rate**, **Num Companies Worked**, **Distance From Home**. 
# - The strongest negative correlations with the target features are: **Total Working Years**, **Job Level**, **Years In Current Role**, and **Monthly Income**.
# - The dataset is **imbalanced** with the majoriy of observations describing Currently Active Employees. <br>
# - Several features (ie columns) are redundant for our analysis, namely: EmployeeCount, EmployeeNumber, StandardHours, and Over18. <br>
# 
# Other observations include: <br>
# > - Single employees show the largest proportion of leavers, compared to Married and Divorced counterparts. <br>
# - About 10% of leavers left when they reach their 2-year anniversary at the company. <br>
# - Loyal employees with higher salaries and more responsbilities show lower proportion of leavers compared to their counterparts. <br>
# - People who live further away from their work show higher proportion of leavers compared to their counterparts.<br>
# - People who travel frequently show higher proportion of leavers compared to their counterparts.<br>
# - People who have to work overtime show higher proportion of leavers compared to their counterparts.<br>
# - Employee who work as Sales Representatives show a significant percentage of Leavers in the submitted dataset.<br>
# - Employees that have already worked at several companies previously (already "bounced" between workplaces) show higher proportion of leavers compared to their counterparts.<br>

# ![title](https://images.unsplash.com/photo-1498409785966-ab341407de6e?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1360&q=80)

# ## Pre-processing Pipeline

# In this section, we undertake data pre-processing steps to prepare the datasets for Machine Learning algorithm implementation.

# ### Encoding

# > Machine Learning algorithms can typically only have numerical values as their predictor variables. Hence Label Encoding becomes necessary as they encode categorical labels with numerical values. To avoid introducing feature importance for categorical features with large numbers of unique values, we will use both Lable Encoding and One-Hot Encoding as shown below.

# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Create a label encoder object
le = LabelEncoder()


# In[ ]:


print(df_HR.shape)
df_HR.head()


# In[ ]:


# Label Encoding will be used for columns with 2 or less unique values
le_count = 0
for col in df_HR.columns[1:]:
    if df_HR[col].dtype == 'object':
        if len(list(df_HR[col].unique())) <= 2:
            le.fit(df_HR[col])
            df_HR[col] = le.transform(df_HR[col])
            le_count += 1
print('{} columns were label encoded.'.format(le_count))


# In[ ]:


# convert rest of categorical variable into dummy
df_HR = pd.get_dummies(df_HR, drop_first=True)


# > The resulting dataframe has **49 columns** for 1,470 employees.

# In[ ]:


print(df_HR.shape)
df_HR.head()


# ### Feature Scaling

# > Feature Scaling using MinMaxScaler essentially shrinks the range such that the range is now between 0 and n. Machine Learning algorithms perform better when input numerical variables fall within a similar scale. In this case, we are scaling between 0 and 5.

# In[ ]:


# import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 5))
HR_col = list(df_HR.columns)
HR_col.remove('Attrition')
for col in HR_col:
    df_HR[col] = df_HR[col].astype(float)
    df_HR[[col]] = scaler.fit_transform(df_HR[[col]])
df_HR['Attrition'] = pd.to_numeric(df_HR['Attrition'], downcast='float')
df_HR.head()


# In[ ]:


print('Size of Full Encoded Dataset: {}'. format(df_HR.shape))


# ### Splitting data into training and testing sets

# > Prior to implementating or applying any Machine Learning algorithms, we must decouple training and testing datasets from our master dataframe.

# In[ ]:


# assign the target to a new dataframe and convert it to a numerical feature
#df_target = df_HR[['Attrition']].copy()
target = df_HR['Attrition'].copy()


# In[ ]:


type(target)


# In[ ]:


# let's remove the target feature and redundant features from the dataset
df_HR.drop(['Attrition', 'EmployeeCount', 'EmployeeNumber','StandardHours', 'Over18'], axis=1, inplace=True)
print('Size of Full dataset is: {}'.format(df_HR.shape))


# In[ ]:


# Since we have class imbalance (i.e. more employees with turnover=0 than turnover=1)
# let's use stratify=y to maintain the same ratio as in the training dataset when splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(df_HR,target,test_size=0.25,random_state=7,stratify=target)  
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# ## Building Machine Learning Models

# ### Baseline Algorithms

# > Let's first use a range of **baseline** algorithms (using out-of-the-box hyper-parameters) before we move on to more sophisticated solutions. The algorithms considered in this section are: **Logistic Regression**, **Random Forest**, **SVM**, **KNN**, **Decision Tree Classifier**, **Gaussian NB**.

# In[ ]:


# selection of algorithms to consider and set performance measure
models = []
models.append(('Logistic Regression', LogisticRegression(solver='liblinear', random_state=7,class_weight='balanced')))
models.append(('Random Forest', RandomForestClassifier(n_estimators=100, random_state=7)))
models.append(('SVM', SVC(gamma='auto', random_state=7)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Decision Tree Classifier',DecisionTreeClassifier(random_state=7)))
models.append(('Gaussian NB', GaussianNB()))


# > Let's evaluate each model in turn and provide accuracy and standard deviation scores

# In[ ]:


acc_results = []
auc_results = []
names = []
# set table to table to populate with performance results
col = ['Algorithm', 'ROC AUC Mean', 'ROC AUC STD','Accuracy Mean', 'Accuracy STD']
df_results = pd.DataFrame(columns=col)
i = 0
# evaluate each model using cross-validation
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7)  # 10-fold cross-validation

    cv_acc_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    cv_auc_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc')

    acc_results.append(cv_acc_results)
    auc_results.append(cv_auc_results)
    names.append(name)
    df_results.loc[i] = [name,round(cv_auc_results.mean()*100, 2),round(cv_auc_results.std()*100, 2),round(cv_acc_results.mean()*100, 2),round(cv_acc_results.std()*100, 2)]
    i += 1
df_results.sort_values(by=['ROC AUC Mean'], ascending=False)


# > **Classification Accuracy** is the number of correct predictions made as a ratio of all predictions made. <br> 
# It is the most common evaluation metric for classification problems. However, it is often **misused** as it is only really suitable when there are an **equal number of observations in each class** and all predictions and prediction errors are equally important. It is not the case in this project, so a different scoring metric may be more suitable.

# In[ ]:


fig = plt.figure(figsize=(15, 7))
fig.suptitle('Algorithm Accuracy Comparison')
ax = fig.add_subplot(111)
plt.boxplot(acc_results)
ax.set_xticklabels(names)
print()


# > **Area under ROC Curve** (or AUC for short) is a performance metric for binary classification problems. <br>
# The AUC represents a **model’s ability to discriminate between positive and negative classes**. An area of 1.0 represents a model that made all predictions perfectly. An area of 0.5 represents a model as good as random.

# In[ ]:


fig = plt.figure(figsize=(15, 7))
fig.suptitle('Algorithm ROC AUC Comparison')
ax = fig.add_subplot(111)
plt.boxplot(auc_results)
ax.set_xticklabels(names)
print()


# > Based on our ROC AUC comparison analysis, **Logistic Regression** and **Random Forest** show the highest mean AUC scores. We will shortlist these two algorithms for further analysis. See below for more details on these two algos.

# **Logistic Regression** is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. Logistic Regression is classification algorithm that is not as sophisticated as the ensemble methods or boosted decision trees method discussed below. Hence, it provides us with a good benchmark. 

# ![title](https://cdn-images-1.medium.com/max/1600/0*vRhSdZ_k4wrP6Bl8.jpg)

# **Random Forest** is a popular and versatile machine learning method that is capable of solving both regression and classification. Random Forest is a brand of Ensemble learning, as it relies on an ensemble of decision trees. It aggregates Classification (or Regression) Trees. A decision tree is composed of a series of decisions that can be used to classify an observation in a dataset.
# 
# Random Forest fits a number of decision tree classifiers on various **sub-samples of the dataset** and use **averaging** to improve the predictive accuracy and control over-fitting. Random Forest can handle a large number of features, and is helpful for estimating which of your variables are important in the underlying data being modeled.

# ![title](https://images.unsplash.com/photo-1441422454217-519d3ee81350?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1489&q=80)

# ### Logistic Regression

# > Let's take a closer look at using the Logistic Regression algorithm. I'll be using 10 fold Cross-Validation to train our Logistic Regression Model and estimate its AUC score.

# In[ ]:


kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression(solver='liblinear',class_weight="balanced",random_state=7)
scoring = 'roc_auc'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("AUC score (STD): %.2f (%.2f)" % (results.mean(), results.std()))


# #### Fine-tuning

# > GridSearchCV allows use to fine-tune hyper-parameters by searching over specified parameter values for an estimator.

# In[ ]:


param_grid = {'C': np.arange(1e-03, 2, 0.01)} # hyper-parameter list to fine-tune
log_gs = GridSearchCV(LogisticRegression(solver='liblinear',class_weight="balanced",random_state=7),iid=True,return_train_score=True,param_grid=param_grid,scoring='roc_auc',cv=10)

log_grid = log_gs.fit(X_train, y_train)
log_opt = log_grid.best_estimator_
results = log_gs.cv_results_

print('='*20)
print("best params: " + str(log_gs.best_estimator_))
print("best params: " + str(log_gs.best_params_))
print('best score:', log_gs.best_score_)
print('='*20)


# > As shown above, the results from GridSearchCV provided us with fine-tuned hyper-parameter using ROC_AUC as the scoring metric.

# #### Evaluation

# In[ ]:


## Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, log_opt.predict(X_test))
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[ ]:


print('Accuracy of Logistic Regression Classifier on test set: {:.2f}'.format(log_opt.score(X_test, y_test)*100))


# > The Confusion matrix provides us with a much more detailed representation of the accuracy score and of what's going on with our labels - we know exactly which/how labels were correctly and incorrectly predicted

# In[ ]:


# Classification report for the optimised Log Regression
log_opt.fit(X_train, y_train)
print(classification_report(y_test, log_opt.predict(X_test)))


# > Instead of getting binary estimated target features (0 or 1), a probability can be associated with the predicted target. <br> The output provides a first index referring to the probability that the data belong to **class 0** (employee not leaving), and the second refers to the probability that the data belong to **class 1** (employee leaving).
# 
# > The resulting AUC score is higher than that best score during the optimisation step. Predicting probabilities of a particular label provides us with a measure of how likely an employee is to leave the company.

# In[ ]:


log_opt.fit(X_train, y_train) # fit optimised model to the training data
probs = log_opt.predict_proba(X_test) # predict probabilities
probs = probs[:, 1] # we will only keep probabilities associated with the employee leaving
logit_roc_auc = roc_auc_score(y_test, probs) # calculate AUC score using test dataset
print('AUC score: %.3f' % logit_roc_auc)


# ### Random Forest Classifier

# > Let's take a closer look at using the Random Forest algorithm. I'll fine-tune the Random Forest algorithm's hyper-parameters by cross-validation against the AUC score.

# #### Fine-tuning

# In[ ]:


rf_classifier = RandomForestClassifier(class_weight = "balanced",random_state=7)
param_grid = {'n_estimators': [50, 75, 100, 125, 150, 175],'min_samples_split':[2,4,6,8,10],'min_samples_leaf': [1, 2, 3, 4],'max_depth': [5, 10, 15, 20, 25]}

grid_obj = GridSearchCV(rf_classifier,iid=True,return_train_score=True,param_grid=param_grid,scoring='roc_auc',cv=10)

grid_fit = grid_obj.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_

print('='*20)
print("best params: " + str(grid_obj.best_estimator_))
print("best params: " + str(grid_obj.best_params_))
print('best score:', grid_obj.best_score_)
print('='*20)


# > Random Forest allows us to know which features are of the most importance in predicting the target feature ("attrition" in this project). Below, we plot features by their importance.

# In[ ]:


importances = rf_opt.feature_importances_
indices = np.argsort(importances)[::-1] # Sort feature importances in descending order
names = [X_train.columns[i] for i in indices] # Rearrange feature names so they match the sorted feature importances
plt.figure(figsize=(15, 7)) # Create plot
plt.title("Feature Importance") # Create plot title
plt.bar(range(X_train.shape[1]), importances[indices]) # Add bars
plt.xticks(range(X_train.shape[1]), names, rotation=90) # Add feature names as x-axis labels
print() # Show plot


# > Random Forest helped us identify the Top 10 most important indicators (ranked in the table below).

# In[ ]:


importances = rf_opt.feature_importances_
df_param_coeff = pd.DataFrame(columns=['Feature', 'Coefficient'])
for i in range(44):
    feat = X_train.columns[i]
    coeff = importances[i]
    df_param_coeff.loc[i] = (feat, coeff)
df_param_coeff.sort_values(by='Coefficient', ascending=False, inplace=True)
df_param_coeff = df_param_coeff.reset_index(drop=True)
df_param_coeff.head(10)


# #### Evaluation

# In[ ]:


## Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, rf_opt.predict(X_test))
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# > The Confusion matrix provides us with a much more detailed representation of the accuracy score and of what's going on with our labels - we know exactly which/how labels were correctly and incorrectly predicted

# In[ ]:


print('Accuracy of RandomForest Regression Classifier on test set: {:.2f}'.format(rf_opt.score(X_test, y_test)*100))


# In[ ]:


# Classification report for the optimised RF Regression
rf_opt.fit(X_train, y_train)
print(classification_report(y_test, rf_opt.predict(X_test)))


# > The resulting AUC score is higher than that best score during the optimisation step. Predicting probabilities of a particular label provides us with a measure of how likely an employee is to leave the company.

# In[ ]:


rf_opt.fit(X_train, y_train) # fit optimised model to the training data
probs = rf_opt.predict_proba(X_test) # predict probabilities
probs = probs[:, 1] # we will only keep probabilities associated with the employee leaving
rf_opt_roc_auc = roc_auc_score(y_test, probs) # calculate AUC score using test dataset
print('AUC score: %.3f' % rf_opt_roc_auc)


# ### ROC Graphs

# > AUC - ROC curve is a performance measurement for classification problem at various thresholds settings. ROC is a probability curve and AUC represents degree or measure of separability. It tells how much model is capable of distinguishing between classes. The green line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top-left corner). <br>

# In[ ]:


# Create ROC Graph
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, log_opt.predict_proba(X_test)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_opt.predict_proba(X_test)[:,1])
plt.figure(figsize=(14, 6))

# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_opt_roc_auc)
# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate' 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
print()


# > As shown above, the fine-tuned Logistic Regression model showed a higher AUC score compared to the Random Forest Classifier. <br>

# ## Concluding Remarks

# ### Risk Category

# As the company generates more data on its employees (on New Joiners and recent Leavers) the algorithm can be re-trained using the additional data and theoritically generate more accurate predictions to identify **high-risk employees** of leaving based on the probabilistic label assigned to each feature variable (i.e. employee) by the algorithm.

# Employees can be assigning a "Risk Category" based on the predicted label such that:
# - **Low-risk** for employees with label < 0.6
# - **Medium-risk** for employees with label between 0.6 and 0.8
# - **High-risk** for employees with label > 0.8 <br>

# ![title](https://images.unsplash.com/photo-1535017584024-2f4bead257df?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80)

# ### Strategic Retention Plan

# - The stronger indicators of people leaving include:
#     - **Monthly Income**: people on higher wages are less likely to leave the company. Hence, efforts should be made to gather information on industry benchmarks in the current local market to determine if the company is providing competitive wages.
#     - **Over Time**: people who work overtime are more likelty to leave the company. Hence efforts  must be taken to appropriately scope projects upfront with adequate support and manpower so as to reduce the use of overtime.
#     - **YearsWithCurrManager**: A large number of leavers leave 6 months after their Current Managers. By using Line Manager details for each employee, one can determine which Manager have experienced the largest numbers of employees resigning over the past year. Several metrics can be used here to determine whether action should be taken with a Line Manager: 
#         - number of employees under managers showing high turnover rates: this would indicate that the organisation's structure may need to be revisit to improve efficiency
#         - number of years the Line Manager has been in a particular position: this may indicate that the employees may need management training or be assigned a mentor (ideally an Executive) in the organisation
#         - patterns in the employees who have resigned: this may indicate recurring patterns in employees leaving in which case action may be taken accordingly.
#     - **Age**: Employees in relatively young age bracket 25-35 are more likely to leave. Hence, efforts should be made to clearly articulate the long-term vision of the company and young employees fit in that vision, as well as provide incentives in the form of clear paths to promotion for instance.
#     - **DistanceFromHome**: Employees who live further from home are more likely to leave the company. Hence, efforts should be made to provide support in the form of company transportation for clusters of employees leaving the same area, or in the form of Transportation Allowance. Initial screening of employees based on their home location is probably not recommended as it would be regarded as a form of discrimination as long as employees make it to work on time every day.
#     - **TotalWorkingYears**: The more experienced employees are less likely to leave. Employees who have between 5-8 years of experience should be identified as potentially having a higher-risk of leaving.
#     - **YearsAtCompany**: Loyal companies are less likely to leave. Employees who hit their two-year anniversary should be identified as potentially having a higher-risk of leaving.

# A strategic **"Retention Plan"** should be drawn for each **Risk Category** group. In addition to the suggested steps for each feature listed above, face-to-face meetings between a HR representative and employees can be initiated for **medium-** and **high-risk employees** to discuss work conditions. Also, a meeting with those employee's Line Manager would allow to discuss the work environment within the team and whether steps can be taken to improve it.

# * **I hope you enjoyed reading this Kernel as much as I had writing it. **

# In[ ]:





