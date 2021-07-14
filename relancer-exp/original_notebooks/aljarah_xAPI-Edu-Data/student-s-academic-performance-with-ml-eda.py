#!/usr/bin/env python
# coding: utf-8

# # Predicting Student's Academic Performance For University

# # Introduction:
# In this small project we are trying to do some new thing.We are try to do make a new mode for this project.We also try to use this project for our real life's problems or project.Very quickly we will try to finish this project and represnt some new thing by this project.Have fun and we will  open to constructive criticisms that will make this project more effective and interesting.
#  ![Imgur](https://i.imgur.com/lsbPelO.jpg)

# # Outline of the Project: 
# <p> <b>1. Exploratory Data Analysis(EDA): </p>
# <a class="link" href="https://www.kaggle.com/harunshimanto" target="_blank">
#                                  Harun-Ur-Rashid(Shimanto)
#                                     </a>
# <p><b>2. Machine Learning Algorithm  applying:</p>
# <a class="link" href="https://www.kaggle.com/harunshimanto" target="_blank">
#                                      Harun-Ur-Rashid(Shimanto)
#                 
# <p><b>3.Deep Learning Algorithm applying:</p>
#                                     

# # 1. Exploratory Data Analysis(EDA): <a class="link" href="https://www.kaggle.com/harunshimanto" target="_blank">Harun-Ur-Rashid(Shimanto)</a>
# <p> <b> 1. What is EDA?
# 
# <b>  <p>In statistics, exploratory data analysis (EDA) is an approach to analyzing data sets to summarize their main characteristics, often with visual methods. A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task. 
# 
# *  <p>You can say that EDA is statisticians way of story telling where you explore data, find patterns and tells insights. Often you have some questions in hand you try to validate those questions by performing EDA.
# [Here is my  article on EDA.](https://hackernoon.com/overview-of-exploratory-data-analysis-with-python-6213e105b00b)

# # Load Import necessary dependencies

# In[ ]:


import numpy as np
import pandas as pd
from subprocess import check_output
print(check_output(["ls", "../../../input/aljarah_xAPI-Edu-Data"]).decode("utf8"))
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, plot_importance


# # Load and Read DataSets

# In[ ]:


data = pd.read_csv("../../../input/aljarah_xAPI-Edu-Data/xAPI-Edu-Data.csv")
# Any results you write to the current directory are saved as output.
data.head()


# <b><p>Attributes</p></b>
# 1 Gender - student's gender (nominal: 'Male' or 'Female’)
# 
# 2 Nationality- student's nationality (nominal:’ Kuwait’,’ Lebanon’,’ Egypt’,’ SaudiArabia’,’ USA’,’ Jordan’,’ Venezuela’,’ Iran’,’ Tunis’,’ Morocco’,’ Syria’,’ Palestine’,’ Iraq’,’ Lybia’)
# 
# 3 Place of birth- student's Place of birth (nominal:’ Kuwait’,’ Lebanon’,’ Egypt’,’ SaudiArabia’,’ USA’,’ Jordan’,’ Venezuela’,’ Iran’,’ Tunis’,’ Morocco’,’ Syria’,’ Palestine’,’ Iraq’,’ Lybia’)
# 
# 4 Educational Stages- educational level student belongs (nominal: ‘lowerlevel’,’MiddleSchool’,’HighSchool’)
# 
# 5 Grade Levels- grade student belongs (nominal: ‘G-01’, ‘G-02’, ‘G-03’, ‘G-04’, ‘G-05’, ‘G-06’, ‘G-07’, ‘G-08’, ‘G-09’, ‘G-10’, ‘G-11’, ‘G-12 ‘)
# 
# 6 Section ID- classroom student belongs (nominal:’A’,’B’,’C’)
# 
# 7 Topic- course topic (nominal:’ English’,’ Spanish’, ‘French’,’ Arabic’,’ IT’,’ Math’,’ Chemistry’, ‘Biology’, ‘Science’,’ History’,’ Quran’,’ Geology’)
# 
# 8 Semester- school year semester (nominal:’ First’,’ Second’)
# 
# 9 Parent responsible for student (nominal:’mom’,’father’)
# 
# 10 Raised hand- how many times the student raises his/her hand on classroom (numeric:0-100)
# 
# 11- Visited resources- how many times the student visits a course content(numeric:0-100)
# 
# 12 Viewing announcements-how many times the student checks the new announcements(numeric:0-100)
# 
# 13 Discussion groups- how many times the student participate on discussion groups (numeric:0-100)
# 
# 14 Parent Answering Survey- parent answered the surveys which are provided from school or not (nominal:’Yes’,’No’)
# 
# 15 Parent School Satisfaction- the Degree of parent satisfaction from school(nominal:’Yes’,’No’)
# 
# 16 Student Absence Days-the number of absence days for each student (nominal: above-7, under-7)

# # DataSets Describe 

# In[ ]:


data.describe()


# # Check DataSets Shape 

# In[ ]:


print(data.shape)


# # Check DataSets Columns

# In[ ]:


data.columns


# # Check Missing Data

# In[ ]:


data.isnull().sum()


# # Gender  Value Counts & Percentage  In Dataset

# In[ ]:


data['gender'].value_counts()


# # NationalITy  Value Counts & Percentage  In Dataset

# In[ ]:


data['NationalITy'].value_counts()


# # PlaceofBirth Value Counts & Percentage In Dataset

# In[ ]:


data['PlaceofBirth'].value_counts()


# *  <b>  Most of these countries are in the middle east(Islamic states), perhaps this explains the gender disparity </b>

# # StageID  Value Counts & Percentage In Dataset

# In[ ]:


data['StageID'].value_counts()


# # GradeID  Value Counts & Percentage  In Dataset

# In[ ]:


data['GradeID'].value_counts()


# # Topic Value Counts & Parcentage In Dataset

# In[ ]:


data['Topic'].value_counts()


# # Semester Value Counts & Parcentage In Dataset

# In[ ]:


data['Semester'].value_counts()


# # Relation Value Counts & Parcentage In Dataset

# In[ ]:


data['Relation'].value_counts()


# # Raisedhands Value Counts & Parcentage In Dataset

# In[ ]:


data['raisedhands'].value_counts()


# # ParentschoolSatisfaction Value Counts & Parcentage In Dataset

# In[ ]:


data['ParentschoolSatisfaction'].value_counts()


# # ParentAnsweringSurvey Value Counts & Parcentage In Dataset

# In[ ]:


data['ParentAnsweringSurvey'].value_counts()


# # StudentAbsenceDays Value Counts & Parcentage In Dataset

# In[ ]:


data['StudentAbsenceDays'].value_counts()


# # Class Value Counts & Parcentage In Dataset

# In[ ]:


data['Class'].value_counts()


# * <p>Girls seem to have performed better than boys
# * Girls had much better attendance than boys</p>

# * <p><b>I'll start with visualizing just the categorical features individually to see what options are included and how each option fares when it comes to count(how many times it appears) and see what I can deduce from that.</p></b>

# In[ ]:


fig, axarr  = plt.subplots(2,2,figsize=(10,10))
sns.countplot(x='Class', data=data, ax=axarr[0,0], order=['L','M','H'])
sns.countplot(x='gender', data=data, ax=axarr[0,1], order=['M','F'])
sns.countplot(x='StageID', data=data, ax=axarr[1,0])
sns.countplot(x='Semester', data=data, ax=axarr[1,1])


# In[ ]:


fig, (axis1, axis2)  = plt.subplots(2, 1,figsize=(10,10))
sns.countplot(x='Topic', data=data, ax=axis1)
sns.countplot(x='NationalITy', data=data, ax=axis2)


# * <p><b>Next I will look at some categorical features in relation to each other, to see what insights that could possibly read</p></b>

# In[ ]:


fig, axarr  = plt.subplots(2,2,figsize=(10,10))
sns.countplot(x='gender', hue='Class', data=data, ax=axarr[0,0], order=['M','F'], hue_order=['L','M','H'])
sns.countplot(x='gender', hue='Relation', data=data, ax=axarr[0,1], order=['M','F'])
sns.countplot(x='gender', hue='StudentAbsenceDays', data=data, ax=axarr[1,0], order=['M','F'])
sns.countplot(x='gender', hue='ParentAnsweringSurvey', data=data, ax=axarr[1,1], order=['M','F'])


# In[ ]:


fig, (axis1, axis2)  = plt.subplots(2, 1,figsize=(10,10))
sns.countplot(x='Topic', hue='gender', data=data, ax=axis1)
sns.countplot(x='NationalITy', hue='gender', data=data, ax=axis2)


# * <p> <b> No apparent gender bias when it comes to subject/topic choices, we cannot conclude that girls performed better because they perhaps took less technical subjects
# * Gender disparity holds even at a country level. May just be as a result of the sampling.</p></b>

# In[ ]:


fig, (axis1, axis2)  = plt.subplots(2, 1,figsize=(10,10))
sns.countplot(x='NationalITy', hue='Relation', data=data, ax=axis1)
sns.countplot(x='NationalITy', hue='StudentAbsenceDays', data=data, ax=axis2)


# 
# ## Now I am moving on to visualizing categorical features with numerical features. 
# 

# In[ ]:


fig, axarr  = plt.subplots(2,2,figsize=(10,10))
sns.barplot(x='Class', y='VisITedResources', data=data, order=['L','M','H'], ax=axarr[0,0])
sns.barplot(x='Class', y='AnnouncementsView', data=data, order=['L','M','H'], ax=axarr[0,1])
sns.barplot(x='Class', y='raisedhands', data=data, order=['L','M','H'], ax=axarr[1,0])
sns.barplot(x='Class', y='Discussion', data=data, order=['L','M','H'], ax=axarr[1,1])


# * <p><b>As expected, those that participated more (higher counts in Discussion, raisedhands, AnnouncementViews, RaisedHands), performed better ...that thing about correlation and causation.</p></b>

# In[ ]:


fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))
sns.barplot(x='gender', y='raisedhands', data=data, ax=axis1)
sns.barplot(x='gender', y='Discussion', data=data, ax=axis2)


# ## There are various other plots that help visualize Categorical vs Numerical data better.

# In[ ]:


fig, (axis1, axis2)  = plt.subplots(1, 2,figsize=(10,5))
sns.boxplot(x='Class', y='Discussion', data=data, order=['L','M','H'], ax=axis1)
sns.boxplot(x='Class', y='VisITedResources', data=data, order=['L','M','H'], ax=axis2)


# * <p> The two plots above tell us that visiting the resources may not be as sure a path to performing well as discussions<p>

# In[ ]:


fig, (axis1, axis2)  = plt.subplots(1, 2,figsize=(10,5))
sns.pointplot(x='Semester', y='VisITedResources', hue='gender', data=data, ax=axis1)
sns.pointplot(x='Semester', y='AnnouncementsView', hue='gender', data=data, ax=axis2)


# * <p>In the case of both visiting resources and viewing announcements, students were more vigilant in the second semester, perhaps that last minute need to boost your final grade.</p>

# ## Moving on to plots to visualize relationships between numerical features.

# In[ ]:


fig, (axis1, axis2)  = plt.subplots(1, 2,figsize=(10,5))
sns.regplot(x='raisedhands', y='VisITedResources', data=data, ax=axis1)
sns.regplot(x='AnnouncementsView', y='Discussion', data=data, ax=axis2)


# ## Gender Comparison With Parents Relationship

# In[ ]:


plot = sns.countplot(x='Class', hue='Relation', data=data, order=['L', 'M', 'H'], palette='Set1')
plot.set(xlabel='Class', ylabel='Count', title='Gender comparison')
print()


#  # 2. Machine Learning Algorithm  applying:<a class="link" href="https://www.kaggle.com/harunshimanto" target="_blank">Harun-Ur-Rashid(Shimanto
#  <p>Now we get to the machine learning section. We will start by encoding our categorical variables and splitting the data into a train and test set.
# 

# ## Label Encoding 

# <p><b>1.Gender Encoding

# In[ ]:


Features = data.drop('gender',axis=1)
Target = data['gender']
label = LabelEncoder()
Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
for col in Cat_Colums:
    Features[col] = label.fit_transform(Features[col])
    


# <p><b>2.Semester Encoding

# In[ ]:


Features = data.drop('Semester',axis=1)
Target = data['Semester']
label = LabelEncoder()
Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
for col in Cat_Colums:
    Features[col] = label.fit_transform(Features[col])


# <p><b>3.ParentAnsweringSurvey Encoding

# In[ ]:


Features = data.drop('ParentAnsweringSurvey',axis=1)
Target = data['ParentAnsweringSurvey']
label = LabelEncoder()
Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
for col in Cat_Colums:
    Features[col] = label.fit_transform(Features[col])


# <p><b>4.Relation Encoding 

# In[ ]:


Features = data.drop('Relation',axis=1)
Target = data['Relation']
label = LabelEncoder()
Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
for col in Cat_Colums:
    Features[col] = label.fit_transform(Features[col])


# <p><b>5.ParentschoolSatisfaction Encoding

# In[ ]:


Features = data.drop('ParentschoolSatisfaction',axis=1)
Target = data['ParentschoolSatisfaction']
label = LabelEncoder()
Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
for col in Cat_Colums:
    Features[col] = label.fit_transform(Features[col])


# <p><b>6.StudentAbsenceDays Encoding

# In[ ]:


Features = data.drop('StudentAbsenceDays',axis=1)
Target = data['StudentAbsenceDays']
label = LabelEncoder()
Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
for col in Cat_Colums:
    Features[col] = label.fit_transform(Features[col])


# <p><b>7.Class Encoding

# In[ ]:


Features = data.drop('Class',axis=1)
Target = data['Class']
label = LabelEncoder()
Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
for col in Cat_Colums:
    Features[col] = label.fit_transform(Features[col])


# ## Test and Train Data Split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=0.2, random_state=52)


# ## Logistic Regression Model

# In[ ]:


Logit_Model = LogisticRegression()
Logit_Model.fit(X_train,y_train)


# <p><b>Logistic Regression's Prediction,Score & Report

# In[ ]:






Prediction = Logit_Model.predict(X_test)
Score = accuracy_score(y_test,Prediction)
Report = classification_report(y_test,Prediction)


# In[ ]:



print(Prediction)


# In[ ]:


print(Score)


# In[ ]:


print(Report)


# ## XGBoost 

# In[ ]:


xgb = XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=100,seed=10)
xgb_pred = xgb.fit(X_train, y_train).predict(X_test)
print (classification_report(y_test,xgb_pred))


# In[ ]:


print(accuracy_score(y_test,xgb_pred))


# In[ ]:


plot_importance(xgb)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20, criterion='entropy')
model.fit(X_train, y_train)
model.score(X_test,y_test)


# <b>Stay With Ours on this project.</b>
# 
# This Kernel is<b> Version 2.5</b>
# > <p>Thanks EveryOne.</p>
#  <p>**Do analysis and stay love Data Science.**</p>
#  
# ** Comming Soon...........**

# In[ ]:





# In[ ]:





# In[ ]:




