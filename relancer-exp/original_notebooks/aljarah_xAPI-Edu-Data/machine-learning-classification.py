#!/usr/bin/env python
# coding: utf-8

# # Introduction 
# 
# 1. [EDA (Exploratory Data Analysis)](#1)
#     1. [Line Plot](#2)
#     1. [Histogram](#3)
#     1. [Scatter Plot](#4)
#     1. [Bar Plot](#5)
#     1. [Point Plot](#6)
#     1. [Count Plot](#7)
#     1. [Pie Chart](#8)
#     1. [Pair Plot](#9)
#    
# 1. [MACHINE LEARNING](#11)
#     1. [Logistic Regression Classification](#12)
#     1. [KNN (K-Nearest Neighbour) Classification](#13)
#     1. [Support Vector Machine( SVM) Classification](#14)
#     1. [Naive Bayes Classification](#15)
#     1. [Decision Tree Classification](#16)
#     1. [Random Forest Classification](#17)
#     1. [Confusion Matrix](#18)
#     
# 1. [Conclusion](#19)  
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# plotly library
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go


# Input data files are available in the "../../../input/aljarah_xAPI-Edu-Data/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../../../input/aljarah_xAPI-Edu-Data"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv("../../../input/aljarah_xAPI-Edu-Data/xAPI-Edu-Data.csv")


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


#heatmap
f,ax=plt.subplots(figsize=(8,8))
print()
print()


# <a id="1"></a> 
# # EDA (Exploratory Data Analysis)

# In[ ]:


data.head()


# <a id="2"></a> 
# # Line Plot

# In[ ]:


# Line Plot
data.raisedhands.plot(kind="line",color="g",label = 'raisedhands',linewidth=1,alpha = 0.5,grid = True,linestyle = ':',figsize=(10,10))
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.legend(loc="upper right")
plt.title('Line Plot')            # title = title of plot
print()


# In[ ]:


plt.subplots(figsize=(10,10))
plt.plot(data.raisedhands[:100],linestyle="-.")
plt.plot(data.VisITedResources[:100],linestyle="-")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Raidehands and VisITedResources Line Plot")
plt.legend(loc="upper right")
print()


# In[ ]:


#subplots
raisedhands=data.raisedhands
VisITedResources=data.VisITedResources

plt.subplots(figsize=(10,10))
plt.subplot(2,1,1)
plt.title("raisedhands-VisITedResources subplot")
plt.plot(raisedhands[:100],color="r",label="raisedhands")
plt.legend()
plt.grid()

plt.subplot(2,1,2)
plt.plot(VisITedResources[:100],color="b",label="VisITedResources")
plt.legend()
plt.grid()

print()


# In[ ]:


# discussion and raisedhands line plot in plotly
# import graph objects as "go"
import plotly.graph_objs as go

# Creating trace1
trace1 = go.Scatter( x = np.arange(0,82), y = data.Discussion, mode = "lines", name = "discussion", marker = dict(color = 'rgba(16, 112, 2, 0.8)'), ) 
# Creating trace2
trace2 = go.Scatter( x =np.arange(0,82) , y = data.raisedhands, mode = "lines", name = "raisedhands", marker = dict(color = 'rgba(80, 26, 80, 0.8)'), ) 
df = [trace1, trace2]
layout = dict(title = 'Discussion and Raisedhands of Students', xaxis= dict(title= 'raisedhands',ticklen= 5,zeroline= False) ) 
fig = dict(data = df, layout = layout)
iplot(fig)


# <a id="3"></a> 
# # Histogram Plot

# In[ ]:


#histogram of raisedhands
data.raisedhands.plot(kind="hist",bins=10,figsize=(10,10),color="b",grid="True")
plt.xlabel("raisedhands")
plt.legend(loc="upper right")
plt.title("raisedhands Histogram")
print()


# In[ ]:


# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
#data.plot(kind="hist",y="raisedhands",bins = 50,range= (0,50),normed = True,ax = axes[0])
#data.plot(kind = "hist",y = "raisedhands",bins = 50,range= (0,50),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
print()


# <a id="4"></a> 
# # Scatter Plot

# In[ ]:


#raidehands vs Discussion scatter plot
plt.subplots(figsize=(10,10))
plt.scatter(data.raisedhands,data.Discussion,color="green")
plt.xlabel("raisedhands")
plt.ylabel("Discussion")
plt.grid()
plt.title("Raidehands vs Discussion Scatter Plot",color="red")
print()


# In[ ]:


#raidehands vs AnnouncementsView scatter plot
color_list1 = ['red' if i=='M' else 'blue' for i in data.gender]
plt.subplots(figsize=(10,10))
plt.scatter(data.raisedhands,data.AnnouncementsView,color=color_list1, alpha=0.8)
plt.xlabel("raisedhands")
plt.ylabel("AnnouncementsView")
plt.grid()
plt.title("Raidehands vs Announcements View Scatter Plot",color="black",fontsize=15)
print()


# # Scatter Plot in Plotly
# 
# * Import graph_objs as *go*
# * Creating traces
#     * x = x axis
#     * y = y axis
#     * mode = type of plot like marker, line or line + markers
#     * name = name of the plots
#     * marker = marker is used with dictionary. 
#         * color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
#     * text = The hover text (hover is curser)
# * data = is a list that we add traces into it
# * layout = it is dictionary.
#     * title = title of layout
#     * x axis = it is dictionary
#         * title = label of x axis
#         * ticklen = length of x axis ticks
#         * zeroline = showing zero line or not
#     * y axis = it is dictionary and same with x axis
# * fig = it includes data and layout
# * iplot() = plots the figure(fig) that is created by data and layout

# In[ ]:


len(data.raisedhands.unique())


# In[ ]:


# raisedhands  in terms of gender

# import graph objects as "go"
import plotly.graph_objs as go

# creating trace1
trace1 =go.Scatter( x = np.arange(0,82), y = data[data.gender=='M'].raisedhands, mode = "markers", name = "male", marker = dict(color = 'rgba(0, 100, 255, 0.8)'), ) 
# creating trace2
trace2 =go.Scatter( x = np.arange(0,82), y = data[data.gender=="F"].raisedhands, mode = "markers", name = "female", marker = dict(color = 'rgba(255, 128, 255, 0.8)'), ) 

df = [trace1, trace2]
layout = dict(title = 'raisedhands', xaxis= dict(title= 'index',ticklen= 5,zeroline= False), yaxis= dict(title= 'Values',ticklen= 5,zeroline= False) ) 
fig = dict(data = df, layout = layout)
iplot(fig)


# In[ ]:


# Discussion  in terms of gender

# import graph objects as "go"
import plotly.graph_objs as go

# creating trace1
trace1 =go.Scatter( x = np.arange(0,82), y = data[data.gender=='M'].Discussion, mode = "markers", name = "male", marker = dict(color = 'rgba(0, 100, 255, 0.8)'), text= data[data.gender=="M"].gender) 
# creating trace2
trace2 =go.Scatter( x = np.arange(0,82), y = data[data.gender=="F"].Discussion, mode = "markers", name = "female", marker = dict(color = 'rgba(200, 50, 150, 0.8)'), text= data[data.gender=="F"].gender) 

df = [trace1, trace2]
layout = dict(title = 'Discussion', xaxis= dict(title= 'index',ticklen= 5,zeroline= False), yaxis= dict(title= 'Values',ticklen= 5,zeroline= False) ) 
fig = dict(data = df, layout = layout)
iplot(fig)


# In[ ]:


# Plotting Scatter Matrix
color_list = ['red' if i=='M' else 'green' for i in data.gender]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'gender'], c=color_list, figsize= [15,15], diagonal='hist', alpha=0.8, s = 200, marker = '.', edgecolor= "black") 

print()


# <a id="5"></a> 
# # Bar Plot

# In[ ]:


# Raisehands Average in terms of Topic

# we will create a data containing averages of the numerical values of our data.
topic_list=list(data.Topic.unique())
rh_av=[]
d_av=[]
aview_av=[]
vr_av=[]
for i in topic_list:
    rh_av.append(sum(data[data["Topic"]==i].raisedhands)/len(data[data["Topic"]==i].raisedhands))
    d_av.append(sum(data[data["Topic"]==i].Discussion)/len(data[data["Topic"]==i].Discussion))
    aview_av.append(sum(data[data["Topic"]==i].AnnouncementsView)/len(data[data["Topic"]==i].AnnouncementsView))
    vr_av.append(sum(data[data["Topic"]==i].VisITedResources)/len(data[data["Topic"]==i].VisITedResources))
data2=pd.DataFrame({"topic":topic_list,"raisedhands_avg":rh_av,"discussion_avg":d_av,"AnnouncementsView_avg":aview_av, "VisITedResources_avg":vr_av})

# we will sort data2 interms of index of raisedhands_avg in ascending order
new_index2 = (data2['raisedhands_avg'].sort_values(ascending=True)).index.values 
sorted_data2 = data2.reindex(new_index2)
sorted_data2.head()

# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2['topic'], y=sorted_data2['raisedhands_avg'])
plt.xticks(rotation= 90)
plt.xlabel('Topics')
plt.ylabel('Raisehands Average')
plt.title("Raisehands Average in terms of Topic")


# In[ ]:


# horizontal bar plot
# Raised hands, Discussion and Announcements View averages acording to topics

f,ax = plt.subplots(figsize = (9,15)) #create a figure of 9x15 .
sns.barplot(x=rh_av,y=topic_list,color='cyan',alpha = 0.5,label='Raised hands' )
sns.barplot(x=d_av,y=topic_list,color='blue',alpha = 0.7,label='Discussion')
sns.barplot(x=aview_av,y=topic_list,color='red',alpha = 0.6,label='Announcements View')

ax.legend(loc='upper right',frameon = True)
ax.set(xlabel='Average ', ylabel='Topics',title = "Average of Numerical Values of Data According to Topics ")


# # Bar Plot in Plotly
# 
# * Import graph_objs as *go*
# * Creating traces
#     * x = x axis
#     * y = y axis
#     * mode = type of plot like marker, line or line + markers
#     * name = name of the plots
#     * marker = marker is used with dictionary. 
#         * color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
#         * line = It is dictionary. line between bars
#             * color = line color around bars
#     * text = The hover text (hover is curser)
# * data = is a list that we add traces into it
# * layout = it is dictionary.
#     * barmode = bar mode of bars like grouped
# * fig = it includes data and layout
# * iplot() = plots the figure(fig) that is created by data and layout

# In[ ]:


# raisehands and discussion average acording to topic

# we will sort data2 interms of index of raisedhands_avg in descending order
new_index3 = (data2['raisedhands_avg'].sort_values(ascending=False)).index.values 
sorted_data3 = data2.reindex(new_index3)

# create trace1 
trace1 = go.Bar( x = sorted_data3.topic, y = sorted_data3.raisedhands_avg, name = "raisedhands average", marker = dict(color = 'rgba(255, 174, 155, 0.5)', line=dict(color='rgb(0,0,0)',width=1.5)), text = sorted_data3.topic) 
# create trace2 
trace2 = go.Bar( x = sorted_data3.topic, y = sorted_data3.discussion_avg, name = "discussion average", marker = dict(color = 'rgba(255, 255, 128, 0.5)', line=dict(color='rgb(0,0,0)',width=1.5)), text = sorted_data3.topic) 
df= [trace1, trace2]
layout = go.Layout(barmode = "group",title= "Discussion and Raisedhands Average of Each Topic")
fig = go.Figure(data = df, layout = layout)
iplot(fig)


# In[ ]:


# raisehands and discussion average acording to PlaceofBirth

#prepare data
place_list=list(data.PlaceofBirth.unique())
rh_av=[]
d_av=[]
aview_av=[]
vr_av=[]
for i in place_list:
    rh_av.append(sum(data[data["PlaceofBirth"]==i].raisedhands)/len(data[data["PlaceofBirth"]==i].raisedhands))
    d_av.append(sum(data[data["PlaceofBirth"]==i].Discussion)/len(data[data["PlaceofBirth"]==i].Discussion))
    aview_av.append(sum(data[data["PlaceofBirth"]==i].AnnouncementsView)/len(data[data["PlaceofBirth"]==i].AnnouncementsView))
    vr_av.append(sum(data[data["PlaceofBirth"]==i].VisITedResources)/len(data[data["PlaceofBirth"]==i].VisITedResources))
data4=pd.DataFrame({"PlaceofBirth":place_list,"raisedhands_avg":rh_av,"discussion_avg":d_av,"AnnouncementsView_avg":aview_av, "VisITedResources_avg":vr_av})

new_index4=data4["raisedhands_avg"].sort_values(ascending=False).index.values
sorted_data4=data4.reindex(new_index4)

# create trace1 
trace1 = go.Bar( x = sorted_data4.PlaceofBirth, y = sorted_data4.raisedhands_avg, name = "raisedhands average", marker = dict(color = 'rgba(200, 125, 200, 0.5)', line=dict(color='rgb(0,0,0)',width=1.5)), text = sorted_data4.PlaceofBirth) 
# create trace2 
trace2 = go.Bar( x = sorted_data4.PlaceofBirth, y = sorted_data4.discussion_avg, name = "discussion average", marker = dict(color = 'rgba(128, 255, 128, 0.5)', line=dict(color='rgb(0,0,0)',width=1.5)), text = sorted_data4.PlaceofBirth) 
df= [trace1, trace2]
layout = go.Layout(barmode = "group",title= "Discussion and Raisedhands Average acording to PlaceofBirth")
fig = go.Figure(data = df, layout = layout)
iplot(fig)


# In[ ]:


trace1 = { 'x': sorted_data4.PlaceofBirth, 'y': sorted_data4.raisedhands_avg, 'name': 'raisedhands average', 'type': 'bar' }; 
trace2 = { 'x': sorted_data4.PlaceofBirth, 'y': sorted_data4.discussion_avg, 'name': 'discussion average', 'type': 'bar' }; 
df = [trace1, trace2];
layout = { 'xaxis': {'title': 'PlaceofBirth'}, 'barmode': 'relative', 'title': 'Raisedhands and Discussion Average Acording to Place of Birth' }; 
fig = go.Figure(data = df, layout = layout)
iplot(fig)


# <a id="6"></a> 
# # Point Plot

# In[ ]:


# Raisedhands vs  Discussion Rate point plot
#normalize the values of discussion_avg and raisedhands_avg
data3=sorted_data2.copy()
data3["raisedhands_avg"]=data3['raisedhands_avg']/max( data3['raisedhands_avg'])
data3["discussion_avg"]=data3['discussion_avg']/max( data3['discussion_avg'])

# visualize
f,ax1 = plt.subplots(figsize =(12,10))
sns.pointplot(x='topic',y='raisedhands_avg',data=data3,color='lime',alpha=0.8)
sns.pointplot(x='topic',y='discussion_avg',data=data3,color='red',alpha=0.8)
plt.text(5,0.50,'Raised hands Average',color='red',fontsize = 17,style = 'italic')
plt.text(5,0.46,'Discussion Average',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('Topics',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Raisedhands vs  Discussion Rate',fontsize = 20,color='blue')
plt.grid()




# In[ ]:


# Raisedhands vs  Discussion Rate point plot acording to place of birth
#normalize the values of discussion_avg and raisedhands_avg
data5=sorted_data4.copy()
data5["raisedhands_avg"]=data5['raisedhands_avg']/max( data5['raisedhands_avg'])
data5["discussion_avg"]=data5['discussion_avg']/max( data5['discussion_avg'])

# visualize
f,ax1 = plt.subplots(figsize =(12,10))
sns.pointplot(x='PlaceofBirth',y='raisedhands_avg',data=data5,color='red',alpha=0.8)
sns.pointplot(x='PlaceofBirth',y='discussion_avg',data=data5,color='blue',alpha=0.8)
plt.text(3,0.30,'Raised hands Average',color='red',fontsize = 17,style = 'italic')
plt.text(3,0.36,'Discussion Average',color='blue',fontsize = 18,style = 'italic')
plt.xlabel('PlaceofBirth',fontsize = 15,color='purple')
plt.ylabel('Values',fontsize = 15,color='purple')
plt.title('Raisedhands vs  Discussion Rate',fontsize = 20,color='purple')
plt.grid()


# <a id="7"></a> 
# # Count Plot

# In[ ]:


data.gender.value_counts()


# In[ ]:


plt.subplots(figsize=(8,5))
sns.countplot(data.gender)
plt.xlabel("gender",fontsize="15")
plt.ylabel("numbers",fontsize="15")
plt.title("Number of Genders in Data", color="red",fontsize="18")
print()


# In[ ]:


#StageID unique values
data.StageID.value_counts()


# In[ ]:


sns.countplot(data.StageID)
plt.xlabel("StageID")
plt.ylabel("numbers")
plt.title("Number of StageID in Data", color="red",fontsize="18")
print()


# <a id="8"></a> 
# # Pie Chart
# 
# * fig: create figures
#     * data: plot type
#         * values: values of plot
#         * labels: labels of plot
#         * name: name of plots
#         * hoverinfo: information in hover
#         * hole: hole width
#         * type: plot type like pie
#     * layout: layout of plot
#         * title: title of layout
#         * annotations: font, showarrow, text, x, y

# In[ ]:


labels=data.StageID.value_counts()
colors=["grey","blue","green"]
explode=[0,0,0]
sizes=data.StageID.value_counts().values

plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title("StageID in Data",color = 'blue',fontsize = 15)
print()


# In[ ]:


# StageID piechart in plotly
pie1_list=data["StageID"].value_counts().values
labels = data["StageID"].value_counts().index
# figure
fig = { "data": [ { "values": pie1_list, "labels": labels, "domain": {"x": [0, .5]}, "name": "StageID", "hoverinfo":"label+percent", "hole": .3, "type": "pie" },], "layout": { "title":"StageID Type", "annotations": [ { "font": { "size": 20}, "showarrow": False, "text": "StageID", "x": 0.20, "y": 1 }, ] } } 
iplot(fig)


# <a id="9"></a> 
# # Pair Plot

# In[ ]:


data.head()


# In[ ]:


# raisedhands and VisITedResources pair plot
print()
print()


# <a id="11"></a> 
# # MACHINE LEARNING
# 
# <a href="https://ibb.co/hgB9j10"><img src="https://i.ibb.co/YNcQMTg/ml1.jpg" alt="ml1" border="0">

# In this part, we will use Machine Learning algoithms in our data. Machine Learning Classification algorithms have the following steps:
# * Split data
# * Fit data
# * Predict Data
# * Find Accuracy

# In[ ]:


data.head()


# We need only numerical features. So create new data containing only numerical values.

# In[ ]:


data_new=data.loc[:,["gender","raisedhands","VisITedResources","AnnouncementsView","Discussion"]]


# We will write 1 for male and 0 for female for classification.

# In[ ]:


data_new.gender=[1 if i=="M" else 0 for i in data_new.gender]


# In[ ]:


data_new.head()


# <a id="12"></a> 
# # Logistic Regression Classification
# 
# * When we talk about binary classification( 0 and 1 outputs) what comes to mind first is logistic regression.
# * Logistic regression is actually a very simple neural network. 

# We need to prepare our data for classificaiton.
# * We will determine x and y values.
#     * y: binary output (0 and 1)
#     * x_data: rest of the data (i.e. features of data except gender)

# In[ ]:


y=data_new.gender.values
x_data=data_new.drop("gender",axis=1)


# In[ ]:


# normalize the values in x_data
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


# * create x_train, y_train, x_test and  y_test arrays with train_test_split method.

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=52)


# In[ ]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
#fit
lr=LogisticRegression()
lr.fit(x_train,y_train)

#accuracy
print("test accuracy is {}".format(lr.score(x_test,y_test)))


# <a id="13"></a> 
# # KNN (K-Nearest Neighbour) Classification
# 1. Choose K value.
# 1. Find the K nearest data points.
# 1. Find the number of data points for each class between K nearest neighbour.
# 1. Identify the class of data or point we tested.

# Assume that we have a graph  and we want to determine the class of black points(i.e. they are in class green or red. )
# <a href="https://ibb.co/NmzLJF6"><img src="https://i.ibb.co/MGLRtgD/2.png" alt="2" border="0">
#     
#  
#  
# 

# 
# * First split data
# * Fit data
# * Predict Data
# * Find Accuracy
# * Find the convenient k value for the highest accuracy.

# In[ ]:


#split data
from sklearn.neighbors import KNeighborsClassifier
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

knn=KNeighborsClassifier(n_neighbors=3)

#fit
knn.fit(x_train,y_train)

#prediction
prediction=knn.predict(x_test)


# In[ ]:


#prediction score (accuracy)
print('KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) 


# In[ ]:


# find the convenient k value for range (1,31)
score_list=[]
train_accuracy=[]
for i in range(1,31):
    knn2=KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    train_accuracy.append(knn2.score(x_train,y_train))
plt.figure(figsize=(15,10))   
plt.plot(range(1,31),score_list,label="testing accuracy",color="blue",linewidth=3)
plt.plot(range(1,31),train_accuracy,label="training accuracy",color="orange",linewidth=3)
plt.xlabel("k values in KNN")
plt.ylabel("accuracy")
plt.title("Accuracy results with respect to k values")
plt.legend()
plt.grid()
print()

print("Maximum value of testing accuracy is {} when k= {}.".format(np.max(score_list),1+score_list.index(np.max(score_list))))


# <a id="14"></a> 
# # Support Vector Machine (SVM) Classification

# In[ ]:


from sklearn.svm import SVC

svm=SVC(random_state=1)
svm.fit(x_train,y_train)
#accuracy
print("accuracy of svm algorithm: ",svm.score(x_test,y_test))


# <a id="15"></a> 
# # Naive Bayes Classification

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)

# test accuracy
print("Accuracy of naive bayees algorithm: ",nb.score(x_test,y_test))


# <a id="16"></a> 
# # Decision Tree Classification
# 
# We have points as seen in the figure and we want to classify these points.
# 
# <a href="https://ibb.co/n8CYzwN"><img src="https://i.ibb.co/G3T8cd4/d11-640x538.jpg" alt="d11-640x538" border="0"></a>
# 
# 

# We will classify these points by using 3 splits.
# 
# <a href="https://ibb.co/y4n4rRk"><img src="https://i.ibb.co/FHbHFWY/d22-690x569.jpg" alt="d22-690x569" border="0"></a>

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

print("Accuracy score for Decision Tree Classification: " ,dt.score(x_test,y_test))


# <a id="17"></a> 
# # Random Forest Classification
# 
# Random Forest divides the train data into n samples, and for each n sample applies Decision Tree algorithm. At he end of n Decision Tree algorithm, it takes the answer which is more.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train,y_train)

print("random forest algorithm accuracy: ",rf.score(x_test,y_test))


# In[ ]:


score_list1=[]
for i in range(100,501,100):
    rf2=RandomForestClassifier(n_estimators=i,random_state=1)
    rf2.fit(x_train,y_train)
    score_list1.append(rf2.score(x_test,y_test))
plt.figure(figsize=(10,10))
plt.plot(range(100,501,100),score_list1)
plt.xlabel("number of estimators")
plt.ylabel("accuracy")
plt.grid()
print()

print("Maximum value of accuracy is {} \nwhen n_estimators= {}.".format(max(score_list1),(1+score_list1.index(max(score_list1)))*100))


# As it seen in the graph, it is convenient to choose n_estimators=100 for the best accuracy result.
# Let's look at between 100 and 130. 

# In[ ]:


score_list2=[]
for i in range(100,131):
    rf3=RandomForestClassifier(n_estimators=i,random_state=1)
    rf3.fit(x_train,y_train)
    score_list2.append(rf3.score(x_test,y_test))
plt.figure(figsize=(10,10))
plt.plot(range(100,131),score_list2)
plt.xlabel("number of estimators")
plt.ylabel("accuracy")
plt.grid()
print()

print("Maximum value of accuracy is {} when number of estimators between 100 and 131 ".format(max(score_list2)))


# Actually, this graph says that, if n_estimators is between 100 and 122, then the value of accuracy is not changing.

# <a id="18"></a> 
# # Confusion Matrix

# Confusion matrix gives the number of true and false predicitons in our classificaiton. It is more reliable than accuracy. 
# <br> Here,
# * y_pred:  results that we predict.
# * y_test: our real values.

# In[ ]:


#Confusion matrix of Random Forest Classf.
y_pred=rf.predict(x_test)
y_true=y_test

#cm
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_true,y_pred)

#cm visualization
f,ax=plt.subplots(figsize=(8,8))
print()
plt.xlabel("predicted value")
plt.ylabel("real value")
print()


# We know that our accuracy is 0.7291, which is the best result, when number of estimators is 100 .  But here we predicted 
# * 21 true for label 0=TN
# * 49 true for label 1=TP
# * 6 wrong for the label 1 =FN ( I predicted 6 label 0 but they are label 1)
# * 20 wrong for label 0= FP ( I predicted 20 label  1 but they are label 0)
# 

# In[ ]:


#Confusion matrix of KNN Classf.
y_pred1=knn.predict(x_test)
y_true=y_test
#cm
cm1=confusion_matrix(y_true,y_pred1)

#cm visualization
f,ax=plt.subplots(figsize=(8,8))
print()
plt.xlabel("predicted value")
plt.ylabel("real value")
print()


# We know that our accuracy is 0.645 when k=3 .  We predicted;
# * 15 true for label 0=TN
# * 47 true for label 1=TP
# * 8 wrong for the label 1 =FN ( I predicted 8 label 0 but they are label 1)
# * 26 wrong for label 0= FP ( I predicted 26 label  1 but they are label 0)

# In[ ]:


#Confusion matrix of Decision Tree Classf.
y_pred2=dt.predict(x_test)
y_true=y_test
#cm
cm2=confusion_matrix(y_true,y_pred2)

#cm visualization
f,ax=plt.subplots(figsize=(8,8))
print()
plt.xlabel("predicted value")
plt.ylabel("real value")
print()


# <a id="19"></a> 
# # Conclusion
# 
# As it seen from confusion matrices, the number of wrong predictions are:
# * KNN Classif: 8, 26
# * Decision Tree Classif:  16, 16
# * Random Forest Classif: 6, 20
# 
# It seems that Random Forest Classification is more effective which can also be seen from accuracy scores. Now lets check this by visualizing the scores.

# In[ ]:


dictionary={"model":["LR","KNN","SVM","NB","DT","RF"],"score":[lr.score(x_test,y_test),knn.score(x_test,y_test),svm.score(x_test,y_test),nb.score(x_test,y_test),dt.score(x_test,y_test),rf.score(x_test,y_test)]}
df1=pd.DataFrame(dictionary)


# In[ ]:


#sort the values of data 
new_index5=df1.score.sort_values(ascending=False).index.values
sorted_data5=df1.reindex(new_index5)

# create trace1 
trace1 = go.Bar( x = sorted_data5.model, y = sorted_data5.score, name = "score", marker = dict(color = 'rgba(200, 125, 200, 0.5)', line=dict(color='rgb(0,0,0)',width=1.5)), text = sorted_data5.model) 
dat = [trace1]
layout = go.Layout(barmode = "group",title= 'Scores of Classifications')
fig = go.Figure(data = dat, layout = layout)
iplot(fig)


# It is clear from above plot that Random Forest Classification is more effective.

# If you have any questions and suggestions, please comment. Your suggestions are very valuable to me.
# I hope you like it.
