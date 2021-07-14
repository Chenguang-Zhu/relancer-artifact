#!/usr/bin/env python
# coding: utf-8

# **Data cleaning&Classification algorithms comparison**

# **In this study, chronic kidney disease was estimated using classification algorithms. You'll find this kernel;**
#             * The codes I think will be useful for data cleaning
#             * How to Handle Missing Data,what did I do?
#             * Data Visualization
#             *  Classification Algorithms
#                 -KNN
#                 -Navie-Bayes
#                 -Logistic Regression
#                 -Decision Tree
#                 -Random Forest
#                 -Support Vector Machine          
#             * Success rate of classification algorithms
#             * Conclusion
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #for confusion matrix
#For data visualization
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Input data files are available in the "../../../input/mansoordaku_ckdisease/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings 
warnings.filterwarnings('ignore')

import os
print(os.listdir("../../../input/mansoordaku_ckdisease"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv("../../../input/mansoordaku_ckdisease/kidney_disease.csv")


# In[ ]:


data.info() #data types and feature names. 
            #I'll change the data types of some parameters in the following codes


# In[ ]:


data.head() #first 5 samples in dataset


# In[ ]:


data.classification.unique() 


# **PREPARE DATA**
# 
# **1) 3 unique values appear in the data set. However, there is no value called  "ckd\t. "  I have written the following code to solve this problem.**

# In[ ]:


data.classification=data.classification.replace("ckd\t","ckd") 


# In[ ]:


data.classification.unique() #problem solved.


# **2) The "id" parameter will not work for classification, so I'm removing this parameter from the data set.**

# In[ ]:


data.drop("id",axis=1,inplace=True) 


# In[ ]:


data.head() #id parameter dropped.


# **3) I changed the target parameter values to 1 and 0 to be able to use the classification algorithms. If the value is "ckd" is 1, if not equal to 0. **

# In[ ]:


data.classification=[1 if each=="ckd" else 0 for each in data.classification]


# In[ ]:


data.head()


# **4) I used the following code to find out how many "NaN" values in which parameter.**

# In[ ]:


data.isnull().sum() 


# 
# **5)Sometimes instead of Nan in the data set, "?" value can be found. To solve this problem ; **df = data [(data! = '?'). all (axis = 1)]** can be used.**
# 

#                        **How to Handle Missing Data,what did I do?**
# 
# Pandas provides the dropna() function that can be used to drop either columns or rows with missing data. We can use dropna() to remove all rows with missing data
# Removing rows with missing values can be too limiting on some predictive modeling problems, an alternative is to impute missing values.
# Imputing refers to using a model to replace missing values.
# 
# There are many options we could consider when replacing a missing value, for example:
# 
#     A constant value that has meaning within the domain, such as 0, distinct from all other values.
#     A value from another randomly selected record.
#     A mean, median or mode value for the column.
#     A value estimated by another predictive model.    
# 
# Pandas provides the fillna() function for replacing missing values with a specific value.
# 
# For example, we can use fillna() to replace missing values with the mean value for each column,
# For example; dataset.fillna(dataset.mean(), inplace=True)

# **6) I can use dropna() to remove all rows with missing data**
#   
# There were 25 parameters of 400 samples before writing this code. After writing this code, there are 25 parameters left in 158 examples.
# 
# The number of samples decreased but the reliability of the model increased.

# In[ ]:



df=data.dropna(axis=0)
print(data.shape)
print(df.shape) 
df.head()


# **7) indexes are not sequential as you can see in the table above. I used the following code to sort indexes.**

# In[ ]:


df.index=range(0,len(df),1)
df.head()


# **8) I corrected some of the parameters.**

# In[ ]:


#you can see that the values have changed.
df.wc=df.wc.replace("\t6200",6200)
df.wc=df.wc.replace("\t8400",8400) 
print(df.loc[11,["wc"]])
print(df.loc[20,["wc"]])


# **9) I'll change the data types of some parameters **

# In[ ]:


df.pcv=df.pcv.astype(int)
df.wc=df.wc.astype(int)
df.rc=df.rc.astype(float)
df.info()


# **10)Keep in mind, the goal in this section is to have all the columns as numeric columns (int or float data type), and containing no missing values. We just dealt with the missing values, so let's now find out the number of columns that are of the object data type and then move on to process them into numeric form.**

# In[ ]:


dtype_object=df.select_dtypes(include=['object'])
dtype_object.head()


# **11)display a sample row to get a better sense of how the values in each column are formatted.**

# In[ ]:


for x in dtype_object.columns:
    print("{} unique values:".format(x),df[x].unique())
    print("*"*20)


# **12)The ordinal values to integers, we can use the pandas DataFrame method replace() to"rbc","pc","pcc","ba","htn","dm","cad","appet","pe" and "ane" to appropriate numeric values**

# In[ ]:


dictonary = { "rbc": { "abnormal":1, "normal": 0, }, "pc":{ "abnormal":1, "normal": 0, }, "pcc":{ "present":1, "notpresent":0, }, "ba":{ "notpresent":0, "present": 1, }, "htn":{ "yes":1, "no": 0, }, "dm":{ "yes":1, "no":0, }, "cad":{ "yes":1, "no": 0, }, "appet":{ "good":1, "poor": 0, }, "pe":{ "yes":1, "no":0, }, "ane":{ "yes":1, "no":0, } } 



# In[ ]:


#We used categorical values as numerical to replace them.
df=df.replace(dictonary)


# In[ ]:


df.head() #All values are numerical.


# **VISUALIZATION**

# In[ ]:


#HEAT MAP #correlation of parameters 
f,ax=plt.subplots(figsize=(15,15))
print()
plt.xticks(rotation=45)
plt.yticks(rotation=45)
print()


# In[ ]:


#box-plot
trace0 = go.Box( y=df.bp, name = 'Bp', marker = dict( color = 'rgb(12, 12, 140)', ) ) 
trace1 = go.Box( y=df.sod, name = 'Sod', marker = dict( color = 'rgb(12, 128, 128)', ) ) 
data = [trace0, trace1]
iplot(data)


# In[ ]:


#Line plot
df2=df.copy()
df2["id"]=range(1,(len(df.ba)+1),1)
df2["df2_bp_norm"]=(df2.bp-np.min(df2.bp))/(np.max(df2.bp)-np.min(df2.bp))
df2["df2_hemo_norm"]=(df2.hemo-np.min(df2.hemo))/(np.max(df2.hemo)-np.min(df2.hemo))
#Line Plot
trace1 = go.Scatter( x = df2.id, y = df2.df2_bp_norm, mode = "lines", name = "Blood Press.", marker = dict(color = 'rgba(16, 112, 2, 0.8)'), text= df.age) 
trace2 = go.Scatter( x = df2.id, y = df2.df2_hemo_norm, mode = "lines+markers", name = "Hemo", marker = dict(color = 'rgba(80, 26, 80, 0.8)'), text= df.age) 
data=[trace1,trace2]
layout=dict(title="Blood Press and Hemoglobin values according the age", xaxis=dict(title="İd",ticklen=5,zeroline=False)) 
fig=dict(data=data,layout=layout)
iplot(fig)


# **CLASSIFICATION ALGORITHMS**

# In[ ]:


score=[] #these variables will be used to show the algorithm name and its successes.
algorithms=[] 


# In[ ]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
y=df["classification"].values
x_data=df.drop(["classification"],axis=1)

#Normalization
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#Preparing the test and training set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.3)

#model and accuracy
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
knn.predict(x_test)
score.append(knn.score(x_test,y_test)*100)
algorithms.append("KNN")
print("KNN accuracy =",knn.score(x_test,y_test)*100)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=knn.predict(x_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
print()
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title(" KNN Confusion Matrix")
print()
#%%


# In[ ]:


#Navie-Bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()

#Training
nb.fit(x_train,y_train)
#Test
score.append(nb.score(x_test,y_test)*100)
algorithms.append("Navie-Bayes")
print("Navie Bayes accuracy =",nb.score(x_test,y_test)*100)

#Confusion Matrix 
from sklearn.metrics import confusion_matrix
y_pred=nb.predict(x_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
print()
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Navie Bayes Confusion Matrix")
print()


# In[ ]:


#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train,y_train)
score.append(rf.score(x_test,y_test)*100)
algorithms.append("Random Forest")
print("Random Forest accuracy =",rf.score(x_test,y_test))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=rf.predict(x_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
print()
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Random Forest Confusion Matrix")
print()


# In[ ]:


#Support Vector Machine
from sklearn.svm import SVC
svm=SVC(random_state=1)
svm.fit(x_train,y_train)
score.append(svm.score(x_test,y_test)*100)
algorithms.append("Support Vector Machine")
print("svm test accuracy =",svm.score(x_test,y_test)*100)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=svm.predict(x_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
print()
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Support Vector Machine Confusion Matrix")
print()


# In[ ]:


#Decision Tree 
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("Decision Tree accuracy:",dt.score(x_test,y_test)*100)
score.append(dt.score(x_test,y_test)*100)
algorithms.append("Decision Tree")

#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=dt.predict(x_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
print()
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Decision Tree Confusion Matrix")
print()


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
score.append(lr.score(x_test,y_test)*100)
algorithms.append("Logistic Regression")
print("test accuracy {}".format(lr.score(x_test,y_test)))
#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=lr.predict(x_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)
#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
print()
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Logistic Regression Confusion Matrix")
print()


# In[ ]:


trace1 = { 'x': algorithms, 'y': score, 'name': 'score', 'type': 'bar' } 


# In[ ]:


data = [trace1];
layout = { 'xaxis': {'title': 'Classification Algorithms'}, 'title': 'Comparison of the accuracy of classification algorithms' }; 
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# **CONCLUSION**
# 
# In this study, there were 400 samples and 26 parameters.However, some samples had no parameter values. For this reason, I prepared the data to use the classification algorithms. I did data visualization work.  I applied classification algorithms and compared success rates with each other.   It was a nice work for me. I hope you like it.
# If you have questions and suggestions, you can comment. Because your questions and suggestions are very valuable for me.
#      
