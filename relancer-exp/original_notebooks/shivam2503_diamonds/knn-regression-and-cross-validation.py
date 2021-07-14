#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pandas as pd
import cufflinks as cf
import sklearn
from sklearn import svm, preprocessing 
import seaborn as sns
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import os


#  ## **[1] Reading Data: **

# In[ ]:


df = pd.read_csv("../../../input/shivam2503_diamonds/diamonds.csv")
df.head()


# **Observations:** 
# 1. Attributes x,y,z define the shape of the diamond. 
# 2. Price is the value we are predicting here, that means the *y* vector. 
# 3. Attributes Cut, clarity, and Color are categorical in nature, for efficient working we can convert them to numeric values. 
# 4. Attribute Unnamed:0 is a additional index value given, as we are storing the data in df, we will use the in pandas index, so this can be removed. 

# In[ ]:


sns.FacetGrid(df, hue = 'cut', size = 6).map(sns.distplot, 'price').add_legend()
plt.plot()


# ## **[2] Preprocessing of Data: **
# ### [2.1] Categorical data to Numeric Data

# In[ ]:


cut_dict = {'Fair' : 1, 'Good' : 2, 'Very Good' : 3, 'Premium' : 4, 'Ideal' : 5}
clarity_dict ={ 'I1' : 1, 'SI2' : 2, 'SI1' : 3, 'VS2' : 4, 'VS1' : 5, 'VVS2' : 6, 'VVS1' : 7 , 'IF' : 8}
color_dict = {'D':7, 'E':6, 'F':5, 'G':4, 'H':3, 'I':2, 'J':1}


# In[ ]:


df['cut'] = df['cut'].map(cut_dict)
df['clarity'] = df['clarity'].map(clarity_dict)
df['color'] = df['color'].map(color_dict)


# ###  [2.2] Dropping additional index attribute. 

# In[ ]:


df = df.drop('Unnamed: 0', axis = 1)
df.head()


# In[ ]:




# In[ ]:


df.isnull().any()


# In[ ]:


df = sklearn.utils.shuffle(df, random_state = 42)
X = df.drop(['price'], axis = 1).values
X = preprocessing.scale(X)
y = df['price'].values
y = preprocessing.scale(y)


# ## **[3] Data modeling: **

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error


# In[ ]:


# Taking 30% data as test and 70% as training data. 
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size = 0.3)


# In[ ]:


# Taking odd integers as K vales so that majority rule can be applied easily. 
neighbors = np.arange(1, 20, 2)
scores = []


# In[ ]:


for k in neighbors:   # running for different K values to know which yields the max accuracy. 
    clf = KNeighborsRegressor(n_neighbors = k,  weights = 'distance', p=1)
    clf.fit(X_tr, y_tr)
    score = cross_val_score(clf, X_tr, y_tr, cv = 10)
    scores.append(score.mean())


# In[ ]:


mse = [1-x for x in scores]


# In[ ]:


trace0 = go.Scatter(y = mse,x = neighbors,mode = 'lines+markers',marker = dict(color = 'rgb(150, 10, 10)'))
layout = go.Layout(title = '',xaxis = dict(title = 'K value',tickmode = 'linear'),yaxis = dict(title = 'CV Error',))
fig = go.Figure(data = [trace0], layout = layout)
iplot(fig, filename='basic-line')


# The K value with least Error will correspond to the best on, as we can see the graph is having a global minima. 

# In[ ]:


optimal_k = neighbors[mse.index(min(mse))]
print("Optimal K: ", optimal_k)


# In[ ]:


# Training the model on Optimal K.
clf_optimal = KNeighborsRegressor(n_neighbors = optimal_k)
clf_optimal.fit(X_tr, y_tr)
y_pred = clf_optimal.predict(X_test)
acc = clf_optimal.score(X_test, y_test)
print("Accuracy: ", acc*100)
print("RMS Error: ", mean_squared_error(y_test, y_pred))


# In[ ]:


trace0 = go.Scatter(y = y_test,x = np.arange(200),mode = 'lines',name = 'Actual Price',marker = dict(color = 'rgb(10, 150, 50)'))

trace1 = go.Scatter(y = y_pred,x = np.arange(200),mode = 'lines',name = 'Predicted Price',line = dict(color = 'rgb(110, 50, 140)',dash = 'dot'))


layout = go.Layout(xaxis = dict(title = 'Index'),yaxis = dict(title = 'Normalized Price'))

figure = go.Figure(data = [trace0, trace1], layout = layout)
iplot(figure)


