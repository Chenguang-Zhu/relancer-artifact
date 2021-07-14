#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

import glob
from datetime import datetime

import sklearn #A Machine Learning library
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# # Acquire Data

# In[ ]:


df = pd.read_csv("../../../input/joniarroba_noshowappointments/KaggleV2-May-2016.csv")


# # Prepare Data

# ## Explore

# In[ ]:


df.info()


# ### Describe data

# In[ ]:


df.describe()


# In[ ]:


df.head(3)


# In[ ]:


print('Alcoholism', df.Alcoholism.unique())
print('Handcap', df.Handcap.unique())
print('Diabetes', df.Diabetes.unique())
print('Hipertension', df.Hipertension.unique())


# ### Data Validation

# In[ ]:


#Checking if there are are any missing values
df.isnull().sum()


# ## Visualize Data(Data Distribution)

# In[ ]:


#Extract date
df['AppointmentDay']=pd.to_datetime(df['AppointmentDay'])
df['ScheduledDay']=pd.to_datetime(df['ScheduledDay'])
df[['AppointmentDay','ScheduledDay']].head(3)


# In[ ]:


df["Month"]=df["AppointmentDay"].dt.month
month = df["Month"].unique()
month


# In[ ]:


g=sns.countplot(x= 'Month',data = df, alpha=0.90)
g=sns.countplot(x= 'Month',data = df, alpha=0.90)
sns.set_color_codes("muted")
g.figure.set_size_inches(12,7)
g.axes.set_title('Total Number of Appointments per Month', fontsize = 28)


# In[ ]:





# In[ ]:


gbarplt=df.groupby(['Gender','No-show'])['No-show'].size()
gbarplt=df.groupby(['Gender','No-show'])['No-show'].size().unstack().plot(kind='bar',stacked=True)
gbarplt.figure.set_size_inches(12,7)
gbarplt.axes.set_title('Appointments per Gender', fontsize = 28)
print(gbarplt)


# In[ ]:


def FormatAge (age):
    if age['Age'] > 0 and age['Age'] < 14:
        return 'Children'
    elif age['Age'] >=14 and age['Age'] < 27:
        return 'Youth'
    elif age['Age'] >=27 and age['Age'] < 64:
        return 'Adults'
    else:
        return 'Senior'


# In[ ]:


df['AgeClass'] = df.apply(FormatAge,axis = 1)


# In[ ]:


barplt = sns.countplot(df['AgeClass'], alpha=0.90)
sns.set_color_codes("muted")
barplt.figure.set_size_inches(12,7)
barplt.axes.set_title('Age Classification', fontsize = 28)


# In[ ]:


gbarplt= sns.countplot(x= 'SMS_received', hue = 'No-show', data = df)
sns.set_color_codes("muted")
gbarplt.figure.set_size_inches(12,7)
gbarplt.axes.set_title('Sms Received vs No-show', fontsize = 28)


# In[ ]:


#Histogram distributuion
bins = np.linspace(0, 120, 20)
hist = sns.distplot(df.Age, bins=bins)
sns.set_color_codes("muted")
hist.set_xlabel('Age', size = 16)
hist.set_ylabel('Frequency', size = 16)
hist.figure.set_size_inches(12,7)
hist.axes.set_title('Age Distribution', fontsize = 28)


# The distribution indicates the presence of outliers for Age less than 0  and greater than 100

# In[ ]:


#Which hospital had most appointments
wordcloud = WordCloud(background_color = 'white', width = 1000, height = 500, collocations=True).generate(str(df['Neighbourhood']))
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.tight_layout(pad=0)
plt.axis('off')
print()


# It appears that the most visited hospital is Jardim Camburi

# In[ ]:


#Remove features which were created as part of the exploration step
df2=df.drop(["AgeClass","Month"],1)


# In[ ]:


#Relationship between variables
correlation = df2.corr()
plt.figure(figsize = (10,10))
corr = sns.heatmap(correlation, vmax=1, square=True, annot=True,cmap='cubehelix')
corr.figure.set_size_inches(15,10)
plt.title("Correlation between variables", size = 28)


# This problem does not present variables that are highly correlated

# #  Pre-Process

# In[ ]:


#Removing outliers
df2=df[df.Age>0]


# #  Model Building

# In[ ]:


#Extracting day from both dates to calcuate lag
df2['Appointment_Day'] = df2['AppointmentDay'].dt.day
df2['Scheduled_Day'] = df2['ScheduledDay'].dt.day
#Calculate the difference between ScheduledDay and AppointmentDay(Lag)
df2['Lag'] = df2['Appointment_Day'] - df2['Scheduled_Day']
df2[['AppointmentDay','Appointment_Day','ScheduledDay','Scheduled_Day','Lag']].head(3)


# In[ ]:


#Verifying if the lag calculated accuratly
df2['Lag'].describe()


# In[ ]:


#Create dummies


# In[ ]:


df3 = pd.get_dummies(df2, columns = ['No-show'])


# In[ ]:


df3.head(3)


# In[ ]:


#Check if dummies were created successfully
df3.info()


# In[ ]:


#Checking for indepency between variables
correlation = df3.corr()
plt.figure(figsize = (10,10))
corr = sns.heatmap(correlation, vmax=1, square=True, annot=True,cmap='cubehelix')
corr.figure.set_size_inches(15,10)
plt.title("Variable independency", size = 28)


# None of the variables of interest are dependent, we can continue to build model

# ## Logistic Regression

# Logistic Regression Assumptions:
#     
#     Target variable is binary
#     Predictive features are interval (continuous) or categorical
#     Features are independent of one another
#     Sample size is adequate – Rule of thumb: 50 records per predictor

# In[ ]:


df3.ix[:,5].head()


# In[ ]:


#Split the data into training and test sets
X = df3.ix[:,(5,12,17)]
y = df3.ix[:,19]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 25)


# In[ ]:


LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)


# In[ ]:


y_pred = LogReg.predict(X_test)
confusion_matrix  = confusion_matrix(y_test, y_pred)
confusion_matrix


# Results reveals that we have 17146 + 4552  correct predictions and 0 + 0 incorrect predictions

# In[ ]:


#Accuracy
print('Accuracy of the logistic regression classifier on the test set: {:.2f}'.format(LogReg.score(X_test, y_test)))


# In[ ]:


print(classification_report(y_test, y_pred))


