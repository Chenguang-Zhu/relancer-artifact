#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../../../input/primaryobjects_voicegender/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../../../input/primaryobjects_voicegender"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Import Standard Python Libraries

# In[ ]:


import io, os, sys, types, time, datetime, math, random, requests, subprocess, tempfile


# ## Packages Import
# 
# These are all the packages we'll be using. Importing individual libraries make it easy for us to use them without having to call the parent libraries.

# In[ ]:


# Data Manipulation 
import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D

# Feature Selection and Encoding
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

# Machine learning 
import sklearn.ensemble as ske
from sklearn import datasets, model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf

# Grid and Random Search
import scipy.stats as st
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Metrics
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

# Managing Warnings 
import warnings
warnings.filterwarnings('ignore')

# Plot the Figures Inline


# ## Data  Loading

# In[ ]:


dataset_raw = pd.read_csv("../../../input/primaryobjects_voicegender/voice.csv")


# ## Data Exploration - Univariate
# 
# When exploring our dataset and its features, we have many options available to us. We can explore each feature individually, or compare pairs of features, finding the correlation between. Let's start with some simple Univariate (one feature) analysis.
# 
# Features can be of multiple types:
# - **Nominal:**  is for mutual exclusive, but not ordered, categories.
# - **Ordinal:** is one where the order matters but not the difference between values.
# - **Interval:** is a measurement where the difference between two values is meaningful.
# - **Ratio:** has all the properties of an interval variable, and also has a clear definition of 0.0.
# 
# There are multiple ways of manipulating each feature type, but for simplicity, we'll define only two feature types:
# - **Numerical:** any feature that contains numeric values.
# - **Categorical:** any feature that contains categories, or text.

# In[ ]:


# Describing all the Numerical Features
dataset_raw.describe()


# In[ ]:


# Describing all the Categorical Features
dataset_raw.describe(include=['O'])


# In[ ]:


# Let's have a quick look at our data
dataset_raw.head()


# In[ ]:


# Let’s plot the distribution of each feature
def plot_distribution(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(dataset.shape[1]) / cols)
    for i, column in enumerate(dataset.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if dataset.dtypes[column] == np.object:
            g = sns.countplot(y=column, data=dataset)
            substrings = [s.get_text()[:18] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)
            plt.xticks(rotation=25)
        else:
            g = sns.distplot(dataset[column])
            plt.xticks(rotation=25)
    
plot_distribution(dataset_raw, cols=3, width=20, height=20, hspace=0.45, wspace=0.5)


# In[ ]:


# How many missing values are there in our dataset?
missingno.matrix(dataset_raw, figsize = (30,5))


# In[ ]:


missingno.bar(dataset_raw, sort='ascending', figsize = (30,5))


# # Feature Cleaning, Engineering, and Imputation
# 
# **Cleaning:**
# To clean our data, we'll need to work with:
# 
# - **Missing values:** Either omit elements from a dataset that contain missing values or impute them (fill them in).
# - **Special values:** Numeric variables are endowed with several formalized special values including ±Inf, NA and NaN. Calculations involving special values often result in special values, and need to be handled/cleaned.
# - **Outliers:** They should be detected, but not necessarily removed. Their inclusion in the analysis is a statistical decision.
# - **Obvious inconsistencies:** A person's age cannot be negative, a man cannot be pregnant and an under-aged person cannot possess a drivers license. Find the inconsistencies and plan for them.
# 
# **Engineering:**
# There are multiple techniques for feature engineering:
# - **Decompose:** Converting 2014-09-20T20:45:40Z into categorical attributes like hour_of_the_day, part_of_day, etc.
# - **Discretization:** We can choose to either discretize some of the continuous variables we have, as some algorithms will perform faster. We are going to do both, and compare the results of the ML algorithms on both discretized and non discretised datasets. We'll call these datasets:
# 
# - dataset_bin => where Continuous variables are Discretised
# - dataset_con => where Continuous variables are Continuous 
# 
# - **Reframe Numerical Quantities:** Changing from grams to kg, and losing detail might be both wanted and efficient for calculation
# - **Feature Crossing:** Creating new features as a combination of existing features. Could be multiplying numerical features, or combining categorical variables. This is a great way to add domain expertise knowledge to the dataset.
#     
# **Imputation:**
# We can impute missing values in a number of different ways:
# - **Hot-Deck:**	The technique then finds the first missing value and uses the cell value immediately prior to the data that are missing to impute the missing value.
# - **Cold-Deck:** Selects donors from another dataset to complete missing data.
# - **Mean-substitution:** Another imputation technique involves replacing any missing value with the mean of that variable for all other cases, which has the benefit of not changing the sample mean for that variable.
# - **Regression:** A regression model is estimated to predict observed values of a variable based on other variables, and that model is then used to impute values in cases where that variable is missing.

# In[ ]:


# To perform our data analysis, let's create new dataframes.
dataset_bin = pd.DataFrame() # To contain our dataframe with our discretised continuous variables 
dataset_con = pd.DataFrame() # To contain our dataframe with our continuous variables 


# ### Feature Label
# 
# This is the feature we are trying to predict. We'll change the string to a binary 0/1. With 1 signifying male.

# In[ ]:


# Let's fix the Class Feature
dataset_raw.loc[dataset_raw['label'] == 'male', 'label'] = 1
dataset_raw.loc[dataset_raw['label'] == 'female', 'label'] = 0

dataset_bin['label'] = dataset_raw['label']
dataset_con['label'] = dataset_raw['label']


# In[ ]:


plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,1)) 
sns.countplot(y="label", data=dataset_bin)


# ### Feature: meanfreq
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['meanfreq'] = pd.cut(dataset_raw['meanfreq'], 10) # discretised 
dataset_con['meanfreq'] = dataset_raw['meanfreq'] # non-discretised


# In[ ]:


plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="meanfreq", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['meanfreq'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['meanfreq'], kde_kws={"label": "female"})


# ### Feature: sd
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['sd'] = pd.cut(dataset_raw['sd'], 10) # discretised 
dataset_con['sd'] = dataset_raw['sd'] # non-discretised


# In[ ]:


plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="sd", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['sd'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['sd'], kde_kws={"label": "female"})


# ### Feature: median
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['median'] = pd.cut(dataset_raw['median'], 10) # discretised 
dataset_con['median'] = dataset_raw['median'] # non-discretised


# In[ ]:


plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="median", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['median'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['median'], kde_kws={"label": "female"})


# ### Feature: Q25
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['Q25'] = pd.cut(dataset_raw['Q25'], 10) # discretised 
dataset_con['Q25'] = dataset_raw['Q25'] # non-discretised

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="Q25", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['Q25'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['Q25'], kde_kws={"label": "female"})


# ### Feature: Q75
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['Q75'] = pd.cut(dataset_raw['Q75'], 10) # discretised 
dataset_con['Q75'] = dataset_raw['Q75'] # non-discretised

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="Q75", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['Q75'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['Q75'], kde_kws={"label": "female"})


# ### Feature: IQR
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['IQR'] = pd.cut(dataset_raw['IQR'], 10) # discretised 
dataset_con['IQR'] = dataset_raw['IQR'] # non-discretised

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="IQR", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['IQR'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['IQR'], kde_kws={"label": "female"})


# ### Feature: skew
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['skew'] = pd.cut(dataset_raw['skew'], 10) # discretised 
dataset_con['skew'] = dataset_raw['skew'] # non-discretised

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="skew", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['skew'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['skew'], kde_kws={"label": "female"})


# ### Feature: kurt
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['kurt'] = pd.cut(dataset_raw['kurt'], 10) # discretised 
dataset_con['kurt'] = dataset_raw['kurt'] # non-discretised

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="kurt", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['kurt'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['kurt'], kde_kws={"label": "female"})


# ### Feature: sp.ent
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['sp.ent'] = pd.cut(dataset_raw['sp.ent'], 10) # discretised 
dataset_con['sp.ent'] = dataset_raw['sp.ent'] # non-discretised

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="sp.ent", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['sp.ent'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['sp.ent'], kde_kws={"label": "female"})


# ### Feature: sfm
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['sfm'] = pd.cut(dataset_raw['sfm'], 10) # discretised 
dataset_con['sfm'] = dataset_raw['sfm'] # non-discretised

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="sfm", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['sfm'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['sfm'], kde_kws={"label": "female"})


# ### Feature: mode
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['mode'] = pd.cut(dataset_raw['mode'], 10) # discretised 
dataset_con['mode'] = dataset_raw['mode'] # non-discretised

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="mode", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['mode'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['mode'], kde_kws={"label": "female"})


# ### Feature: centroid
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['centroid'] = pd.cut(dataset_raw['centroid'], 10) # discretised 
dataset_con['centroid'] = dataset_raw['centroid'] # non-discretised

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="centroid", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['centroid'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['centroid'], kde_kws={"label": "female"})


# ### Feature: meanfun
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['meanfun'] = pd.cut(dataset_raw['meanfun'], 10) # discretised 
dataset_con['meanfun'] = dataset_raw['meanfun'] # non-discretised

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="meanfun", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['meanfun'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['meanfun'], kde_kws={"label": "female"})


# ### Feature: minfun
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['minfun'] = pd.cut(dataset_raw['minfun'], 10) # discretised 
dataset_con['minfun'] = dataset_raw['minfun'] # non-discretised

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="minfun", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['minfun'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['minfun'], kde_kws={"label": "female"})


# ### Feature: maxfun
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['maxfun'] = pd.cut(dataset_raw['maxfun'], 10) # discretised 
dataset_con['maxfun'] = dataset_raw['maxfun'] # non-discretised

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="maxfun", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['maxfun'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['maxfun'], kde_kws={"label": "female"})


# ### Feature: meandom
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['meandom'] = pd.cut(dataset_raw['meandom'], 10) # discretised 
dataset_con['meandom'] = dataset_raw['meandom'] # non-discretised

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="meandom", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['meandom'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['meandom'], kde_kws={"label": "female"})


# ### Feature: mindom
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['mindom'] = pd.cut(dataset_raw['mindom'], 10) # discretised 
dataset_con['mindom'] = dataset_raw['mindom'] # non-discretised

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="mindom", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['mindom'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['mindom'], kde_kws={"label": "female"})


# ### Feature: maxdom
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['maxdom'] = pd.cut(dataset_raw['maxdom'], 10) # discretised 
dataset_con['maxdom'] = dataset_raw['maxdom'] # non-discretised

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="maxdom", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['maxdom'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['maxdom'], kde_kws={"label": "female"})


# ### Feature: dfrange
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['dfrange'] = pd.cut(dataset_raw['dfrange'], 10) # discretised 
dataset_con['dfrange'] = dataset_raw['dfrange'] # non-discretised

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="dfrange", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['dfrange'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['dfrange'], kde_kws={"label": "female"})


# ### Feature: modindx
# 
# We will use the Pandas Cut function to bin the data in equally sized buckets. We will also add our original feature to the dataset_con dataframe.

# In[ ]:


dataset_bin['modindx'] = pd.cut(dataset_raw['modindx'], 10) # discretised 
dataset_con['modindx'] = dataset_raw['modindx'] # non-discretised

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="modindx", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['label'] == 1]['modindx'], kde_kws={"label": "male"})
sns.distplot(dataset_con.loc[dataset_con['label'] == 0]['modindx'], kde_kws={"label": "female"})


# ## Bi-variate Analysis
# 
# So far, we have analised all features individually. Let's now start combining some of these features together to obtain further insight into the interactions between them.

# In[ ]:


# Interaction between pairs of features.
#todo select some features


# ## Feature Encoding
# 
# Remember that Machine Learning algorithms perform Linear Algebra on Matrices, which means all features need have numeric values. The process of converting Categorical Features into values is called Encoding. Let's perform both One-Hot and Label encoding.

# In[ ]:


# One Hot Encodes all labels before Machine Learning
one_hot_cols = dataset_bin.columns.tolist()
one_hot_cols.remove('label')
dataset_bin_enc = pd.get_dummies(dataset_bin, columns=one_hot_cols)

dataset_bin_enc.head()


# In[ ]:


# Label Encode all labels
dataset_con_enc = dataset_con.apply(LabelEncoder().fit_transform)

dataset_con_enc.head()


# ## Feature Reduction / Selection
# 
# Once we have our features ready to use, we might find that the number of features available is too large to be run in a reasonable timeframe by our machine learning algorithms. There's a number of options available to us for feature reduction and feature selection.
# 
# - **Dimensionality Reduction:**
#     - **Principal Component Analysis (PCA):** Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components.
#     - **Singular Value Decomposition (SVD):** SVD is a factorization of a real or complex matrix. It is the generalization of the eigendecomposition of a positive semidefinite normal matrix (for example, a symmetric matrix with positive eigenvalues) to any m×n  matrix via an extension of the polar decomposition. It has many useful applications in signal processing and statistics.
# 
# 
# - **Feature Importance/Relevance:**
#     - **Filter Methods:** Filter type methods select features based only on general metrics like the correlation with the variable to predict. Filter methods suppress the least interesting variables. The other variables will be part of a classification or a regression model used to classify or to predict data. These methods are particularly effective in computation time and robust to overfitting.
#     - **Wrapper Methods:** Wrapper methods evaluate subsets of variables which allows, unlike filter approaches, to detect the possible interactions between variables. The two main disadvantages of these methods are : The increasing overfitting risk when the number of observations is insufficient. AND. The significant computation time when the number of variables is large.
#     - **Embedded Methods:** Embedded methods try to combine the advantages of both previous methods. A learning algorithm takes advantage of its own variable selection process and performs feature selection and classification simultaneously.

# ### Feature Correlation
# 
# Correlation ia s measure of how much two random variables change together. Features should be uncorrelated with each other and highly correlated to the feature we’re trying to predict.

# In[ ]:


# Create a correlation plot of both datasets.
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(25,10)) 

plt.subplot(1, 2, 1)
# Generate a mask for the upper triangle
mask = np.zeros_like(dataset_bin_enc.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.subplot(1, 2, 2)
mask = np.zeros_like(dataset_con_enc.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


# ### Feature Importance
# 
# Random forest consists of a number of decision trees. Every node in the decision trees is a condition on a single feature, designed to split the dataset into two so that similar response values end up in the same set. The measure based on which the (locally) optimal condition is chosen is called impurity. When training a tree, it can be computed how much each feature decreases the weighted impurity in a tree. For a forest, the impurity decrease from each feature can be averaged and the features are ranked according to this measure. This is the feature importance measure exposed in sklearn’s Random Forest implementations.

# In[ ]:


# Using Random Forest to gain an insight on Feature Importance
clf = RandomForestClassifier()
clf.fit(dataset_con_enc.drop('label', axis=1), dataset_con_enc['label'])

plt.style.use('seaborn-whitegrid')
importance = clf.feature_importances_
importance = pd.DataFrame(importance, index=dataset_con_enc.drop('label', axis=1).columns, columns=["Importance"])
importance.sort_values(by='Importance', ascending=True).plot(kind='barh', figsize=(20,len(importance)/2))


# ### PCA
# 
# Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components.
# 
# We can use PCA to reduce the number of features to use in our ML algorithms, and graphing the variance gives us an idea of how many features we really need to represent our dataset fully.

# In[ ]:


# Calculating PCA for both datasets, and graphing the Variance for each feature, per dataset
std_scale = preprocessing.StandardScaler().fit(dataset_bin_enc.drop('label', axis=1))
X = std_scale.transform(dataset_bin_enc.drop('label', axis=1))
pca1 = PCA(n_components=len(dataset_bin_enc.columns)-1)
fit1 = pca1.fit(X)

std_scale = preprocessing.StandardScaler().fit(dataset_con_enc.drop('label', axis=1))
X = std_scale.transform(dataset_con_enc.drop('label', axis=1))
pca2 = PCA(n_components=len(dataset_con_enc.columns)-2)
fit2 = pca2.fit(X)

# Graphing the variance per feature
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(25,7)) 

plt.subplot(1, 2, 1)
plt.xlabel('PCA Feature')
plt.ylabel('Variance')
plt.title('PCA for Discretised Dataset')
plt.bar(range(0, fit1.explained_variance_ratio_.size), fit1.explained_variance_ratio_)

plt.subplot(1, 2, 2)
plt.xlabel('PCA Feature')
plt.ylabel('Variance')
plt.title('PCA for Continuous Dataset')
plt.bar(range(0, fit2.explained_variance_ratio_.size), fit2.explained_variance_ratio_)


# In[ ]:


# PCA's components graphed in 2D and 3D
# Apply Scaling 
std_scale = preprocessing.StandardScaler().fit(dataset_con_enc.drop('label', axis=1))
X = std_scale.transform(dataset_con_enc.drop('label', axis=1))
y = dataset_con_enc['label']

# Formatting
target_names = [0,1]
colors = ['navy','darkorange']
lw = 2
alpha = 0.3
# 2 Components PCA
plt.style.use('seaborn-whitegrid')
plt.figure(2, figsize=(20, 8))

plt.subplot(1, 2, 1)
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1],color=color,alpha=alpha,lw=lw,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('First two PCA directions')

# 3 Components PCA
ax = plt.subplot(1, 2, 2, projection='3d')

pca = PCA(n_components=3)
X_reduced = pca.fit(X).transform(X)
for color, i, target_name in zip(colors, [0, 1], target_names):
    ax.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1], X_reduced[y == i, 2],color=color,alpha=alpha,lw=lw,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.set_ylabel("2nd eigenvector")
ax.set_zlabel("3rd eigenvector")

# rotate the axes
ax.view_init(30, 10)


# ### Recursive Feature Elimination
# 
# Feature ranking with recursive feature elimination and cross-validated selection of the best number of features.

# In[ ]:


# Calculating RFE for non-discretised dataset, and graphing the Importance for each feature, per dataset
selector1 = RFECV(LogisticRegression(), step=1, cv=5, n_jobs=1)
selector1 = selector1.fit(dataset_con_enc.drop('label', axis=1).values, dataset_con_enc['label'].values)
print("Feature Ranking For Non-Discretised: %s" % selector1.ranking_)
print("Optimal number of features : %d" % selector1.n_features_)
# Plot number of features VS. cross-validation scores
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(20,5)) 
plt.xlabel("Number of features selected - Non-Discretised")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(selector1.grid_scores_) + 1), selector1.grid_scores_)

# Feature space could be subsetted like so:
dataset_con_enc = dataset_con_enc[dataset_con_enc.columns[np.insert(selector1.support_, 0, True)]]


# ## Selecting Dataset
# 
# We now have two datasets to choose from to apply our ML algorithms. The one-hot-encoded, and the label-encoded. For now, we have decided not to use feature reduction or selection algorithms.

# In[ ]:


selected_dataset = dataset_con_enc


# In[ ]:


selected_dataset.head(2)


# ## Splitting Data into Training and Testing Datasets
# 
# We need to split the data back into the training and testing datasets. Remember we joined both right at the beginning.

# In[ ]:


from sklearn.model_selection import train_test_split
X = selected_dataset.drop(['label'], axis=1)
y = selected_dataset['label'].astype('int64')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# ## Machine Learning Algorithms

# ### Data Review
# 
# Let's take one last peek at our data before we start running the Machine Learning algorithms.

# In[ ]:


X_train.shape


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# In[ ]:


random.seed(1)


# ### Algorithms
# 
# From here, we will be running the following algorithms.
# 
# - KNN
# - Logistic Regression
# - Random Forest
# - Naive Bayes
# - Stochastic Gradient Decent
# - Linear SVC
# - Decision Tree
# - Gradient Boosted Trees
# 
# Because there's a great deal of repetitiveness on the code for each, we'll create a custom function to analyse this.
# 
# For some algorithms, we have also chosen to run a Random Hyperparameter search, to select the best hyperparameters for a given algorithm.

# In[ ]:


# calculate the fpr and tpr for all thresholds of the classification
def plot_roc_curve(y_test, preds):
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    print()


# In[ ]:


# Function that runs the requested algorithm and returns the accuracy metrics
def fit_ml_algo(algo, X_train, y_train, X_test, cv):
    # One Pass
    model = algo.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    if (isinstance(algo, (LogisticRegression,KNeighborsClassifier,GaussianNB,DecisionTreeClassifier,RandomForestClassifier,GradientBoostingClassifier))):
        probs = model.predict_proba(X_test)[:,1]
    else:
        probs = "Not Available"
    acc = round(model.score(X_test, y_test) * 100, 2) 
    # CV 
    train_pred = model_selection.cross_val_predict(algo,X_train,y_train,cv=cv,n_jobs = -1)
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    return train_pred, test_pred, acc, acc_cv, probs


# In[ ]:


# Logistic Regression - Random Search for Hyperparameters

# Utility function to report best scores
def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
# Specify parameters and distributions to sample from
param_dist = {'penalty': ['l2', 'l1'],'class_weight': [None, 'balanced'],'C': np.logspace(-20, 20, 10000),'intercept_scaling': np.logspace(-20, 20, 10000)}

# Run Randomized Search
n_iter_search = 10
lrc = LogisticRegression()
random_search = RandomizedSearchCV(lrc,param_distributions=param_dist,n_iter=n_iter_search)

start = time.time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time.time() - start), n_iter_search))
report(random_search.cv_results_)


# In[ ]:


# Logistic Regression
start_time = time.time()
train_pred_log, test_pred_log, acc_log, acc_cv_log, probs_log = fit_ml_algo(LogisticRegression(),X_train,y_train,X_test,10)
log_time = (time.time() - start_time)
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))


# In[ ]:


print(metrics.classification_report(y_train, train_pred_log)) 


# In[ ]:


print(metrics.classification_report(y_test, test_pred_log)) 


# In[ ]:


plot_roc_curve(y_test, probs_log)


# In[ ]:


# k-Nearest Neighbors
start_time = time.time()
train_pred_knn, test_pred_knn, acc_knn, acc_cv_knn, probs_knn = fit_ml_algo(KNeighborsClassifier(n_neighbors = 3),X_train,y_train,X_test,10)
knn_time = (time.time() - start_time)
print("Accuracy: %s" % acc_knn)
print("Accuracy CV 10-Fold: %s" % acc_cv_knn)
print("Running Time: %s" % datetime.timedelta(seconds=knn_time))


# In[ ]:


print(metrics.classification_report(y_train, train_pred_knn)) 


# In[ ]:


print(metrics.classification_report(y_test, test_pred_knn)) 


# In[ ]:


plot_roc_curve(y_test, probs_knn)


# In[ ]:


# Gaussian Naive Bayes
start_time = time.time()
train_pred_gaussian, test_pred_gaussian, acc_gaussian, acc_cv_gaussian, probs_gau = fit_ml_algo(GaussianNB(),X_train,y_train,X_test,10)
gaussian_time = (time.time() - start_time)
print("Accuracy: %s" % acc_gaussian)
print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)
print("Running Time: %s" % datetime.timedelta(seconds=gaussian_time))


# In[ ]:


print(metrics.classification_report(y_train, train_pred_gaussian)) 


# In[ ]:


print(metrics.classification_report(y_test, test_pred_gaussian))


# In[ ]:


plot_roc_curve(y_test, probs_gau)


# In[ ]:


# Linear SVC
start_time = time.time()
train_pred_svc, test_pred_svc, acc_linear_svc, acc_cv_linear_svc, _ = fit_ml_algo(LinearSVC(),X_train,y_train,X_test,10)
linear_svc_time = (time.time() - start_time)
print("Accuracy: %s" % acc_linear_svc)
print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)
print("Running Time: %s" % datetime.timedelta(seconds=linear_svc_time))


# In[ ]:


print(metrics.classification_report(y_train, train_pred_svc)) 


# In[ ]:


print(metrics.classification_report(y_test, test_pred_svc)) 


# In[ ]:


# Stochastic Gradient Descent
start_time = time.time()
train_pred_sgd, test_pred_sgd, acc_sgd, acc_cv_sgd, _ = fit_ml_algo(SGDClassifier(),X_train,y_train,X_test,10)
sgd_time = (time.time() - start_time)
print("Accuracy: %s" % acc_sgd)
print("Accuracy CV 10-Fold: %s" % acc_cv_sgd)
print("Running Time: %s" % datetime.timedelta(seconds=sgd_time))


# In[ ]:


print(metrics.classification_report(y_train, train_pred_sgd)) 


# In[ ]:


print(metrics.classification_report(y_test, test_pred_sgd)) 


# In[ ]:


# Decision Tree Classifier
start_time = time.time()
train_pred_dt, test_pred_dt, acc_dt, acc_cv_dt, probs_dt = fit_ml_algo(DecisionTreeClassifier(),X_train,y_train,X_test,10)
dt_time = (time.time() - start_time)
print("Accuracy: %s" % acc_dt)
print("Accuracy CV 10-Fold: %s" % acc_cv_dt)
print("Running Time: %s" % datetime.timedelta(seconds=dt_time))


# In[ ]:


print(metrics.classification_report(y_train, train_pred_dt)) 


# In[ ]:


print(metrics.classification_report(y_test, test_pred_dt)) 


# In[ ]:


plot_roc_curve(y_test, probs_dt)


# In[ ]:


# Random Forest Classifier - Random Search for Hyperparameters

# Utility function to report best scores
def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
# Specify parameters and distributions to sample from
param_dist = {"max_depth": [10, None],"max_features": sp_randint(1, 7),"min_samples_split": sp_randint(2, 20),"min_samples_leaf": sp_randint(1, 11),"bootstrap": [True, False],"criterion": ["gini", "entropy"]}

# Run Randomized Search
n_iter_search = 10
rfc = RandomForestClassifier(n_estimators=10)
random_search = RandomizedSearchCV(rfc,param_distributions=param_dist,n_iter=n_iter_search)

start = time.time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time.time() - start), n_iter_search))
report(random_search.cv_results_)


# In[ ]:


# Random Forest Classifier
start_time = time.time()
rfc = RandomForestClassifier(n_estimators=10,min_samples_leaf=2,min_samples_split=17,criterion='gini',max_features=8)
train_pred_rf, test_pred_rf, acc_rf, acc_cv_rf, probs_rf = fit_ml_algo(rfc,X_train,y_train,X_test,10)
rf_time = (time.time() - start_time)
print("Accuracy: %s" % acc_rf)
print("Accuracy CV 10-Fold: %s" % acc_cv_rf)
print("Running Time: %s" % datetime.timedelta(seconds=rf_time))


# In[ ]:


print(metrics.classification_report(y_train, train_pred_rf)) 


# In[ ]:


print(metrics.classification_report(y_test, test_pred_rf)) 


# In[ ]:


plot_roc_curve(y_test, probs_rf)


# In[ ]:


# Gradient Boosting Trees
start_time = time.time()
train_pred_gbt, test_pred_gbt, acc_gbt, acc_cv_gbt, probs_gbt = fit_ml_algo(GradientBoostingClassifier(),X_train,y_train,X_test,10)
gbt_time = (time.time() - start_time)
print("Accuracy: %s" % acc_gbt)
print("Accuracy CV 10-Fold: %s" % acc_cv_gbt)
print("Running Time: %s" % datetime.timedelta(seconds=gbt_time))


# In[ ]:


print(metrics.classification_report(y_train, train_pred_gbt)) 


# In[ ]:


print(metrics.classification_report(y_test, test_pred_gbt)) 


# In[ ]:


plot_roc_curve(y_test, probs_gbt)


# ## Ranking Results
# 
# Let's rank the results for all the algorithms we have used

# In[ ]:


models = pd.DataFrame({'Model': ['KNN', 'Logistic Regression','Random Forest', 'Naive Bayes','Stochastic Gradient Decent', 'Linear SVC','Decision Tree', 'Gradient Boosting Trees'],'Score':[acc_knn,acc_log,acc_rf,acc_gaussian,acc_sgd,acc_linear_svc,acc_dt,acc_gbt]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


models = pd.DataFrame({'Model': ['KNN', 'Logistic Regression','Random Forest', 'Naive Bayes','Stochastic Gradient Decent', 'Linear SVC','Decision Tree', 'Gradient Boosting Trees'],'Score':[acc_cv_knn,acc_cv_log,acc_cv_rf,acc_cv_gaussian,acc_cv_sgd,acc_cv_linear_svc,acc_cv_dt,acc_cv_gbt]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(10,10)) 

models =['KNN','Logistic Regression','Random Forest','Naive Bayes','Decision Tree','Gradient Boosting Trees']
probs =[probs_knn,probs_log,probs_rf,probs_gau,probs_dt,probs_gbt]
colors =['blue','green','red','cyan','magenta','yellow',]
    
plt.title('Receiver Operating Characteristic')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

def plot_roc_curves(y_test, prob, model):
    fpr, tpr, threshold = metrics.roc_curve(y_test, prob)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label = model + ' AUC = %0.2f' % roc_auc, color=colors[i])
    plt.legend(loc = 'lower right')
    
for i, model in list(enumerate(models)):
    plot_roc_curves(y_test, probs[i], models[i])
    
print()


# ## Tensorflow - Logistic Regression
# 
# Reference: https://www.tensorflow.org/tutorials/wide

# ### Converting Data into Tensors
# When building a TF.Learn model, the input data is specified by means of an Input Builder function. This builder function will not be called until it is later passed to TF.Learn methods such as fit and evaluate. The purpose of this function is to construct the input data, which is represented in the form of tf.Tensors or tf.SparseTensors. In more detail, the Input Builder function returns the following as a pair:
# 
# - feature_cols: A dict from feature column names to Tensors or SparseTensors.
# - label: A Tensor containing the label column.

# In[ ]:


df1 = pd.DataFrame(dataset_con.dtypes, columns=['Continuous Type'])
df2 = pd.DataFrame(dataset_bin.dtypes, columns=['Discretised Type'])
pd.concat([df1, df2], axis=1).transpose()


# In[ ]:


# Selecting the Continuous Dataset
LABEL_COLUMN = "label"
dataset_con[LABEL_COLUMN] = dataset_con["label"].astype(int)

CONTINUOUS_COLUMNS = dataset_con.select_dtypes(include=[np.number]).columns.tolist()
CATEGORICAL_COLUMNS =[]


# In[ ]:


# Missing Values
missingno.matrix(dataset_con, figsize = (30,5))


# In[ ]:


# Splitting the Training and Test data sets
train = dataset_con.loc[0:2900,:]
test = dataset_con.loc[2900:,:]


# In[ ]:


# Dropping rows with Missing Values
train = train.dropna(axis=0)
test = test.dropna(axis=0)


# In[ ]:


# Coverting Dataframes into Tensors
def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(indices=[[i, 0] for i in range(df[k].size)],values=df[k].values,dense_shape=[df[k].size, 1]) for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  d = continuous_cols.copy()
  d.update(categorical_cols)
  feature_cols = d
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(train)

def eval_input_fn():
  return input_fn(test)


# ### Base Categorical Feature Columns
# To define a feature column for a categorical feature, we can create a SparseColumn using the TF.Learn API. If you know the set of all possible feature values of a column and there are only a few of them, you can use sparse_column_with_keys. Each key in the list will get assigned an auto-incremental ID starting from 0. If we don't know the set of possible values in advance, we can use sparse_column_with_hash_bucket instead:

# In[ ]:


# Listing categorical classes for reference
train.select_dtypes(include=[np.object]).columns.tolist()


# ### Base Continuous Feature Columns
# Similarly, we can define a RealValuedColumn for each continuous feature column that we want to use in the model:

# In[ ]:


train.select_dtypes(include=[np.number]).columns.tolist()


# In[ ]:


#IQR	sfm	meanfun	minfun	maxfun	mindom	maxdom	dfrange
IQR = tf.contrib.layers.real_valued_column("IQR")
sfm = tf.contrib.layers.real_valued_column("sfm")
meanfun = tf.contrib.layers.real_valued_column("meanfun")
minfun = tf.contrib.layers.real_valued_column("minfun")
maxfun = tf.contrib.layers.real_valued_column("maxfun")
mindom = tf.contrib.layers.real_valued_column("mindom")
maxdom = tf.contrib.layers.real_valued_column("maxdom")
dfrange = tf.contrib.layers.real_valued_column("dfrange")


# ### Defining The Logistic Regression Model
# 
# After processing the input data and defining all the feature columns, we're now ready to put them all together and build a Logistic Regression model.

# In[ ]:


model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=[IQR,sfm,meanfun,minfun,maxfun,mindom,maxdom,dfrange],model_dir=model_dir)


# ### Training and Evaluating Our Model
# 
# After adding all the features to the model, now let's look at how to actually train the model. Training a model is just a one-liner using the TF.Learn API:

# In[ ]:


m.fit(input_fn=train_input_fn, steps=200)


# In[ ]:


results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))


# In[ ]:





