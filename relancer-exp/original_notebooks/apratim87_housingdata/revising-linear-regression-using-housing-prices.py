#!/usr/bin/env python
# coding: utf-8

# Just a little notebook revising some of the principles of linear regression, and practising how to execute it in python using the 1978 Boston Housing dataset (original dataset can be found at https://archive.ics.uci.edu/ml/datasets/Housing).
# 
# First step is to read in the data and explore its features.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import Image
from IPython.core.display import HTML


# Input data files are available in the "../../../input/apratim87_housingdata/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../../../input/apratim87_housingdata/"))

# Any results you write to the current directory are saved as output.


# In[2]:


# import the data into a pandas dataframe
housing_df = pd.read_csv("../../../input/apratim87_housingdata/housingdata.csv", header=None)
# inspect the dataframe
housing_df.head()


# **Column name info:**
# - CRIM per capita crime rate by town
# 
# - ZN proportion of residential land zoned for lots over 25,000 sq.ft.
# 
# - INDUS proportion of non-retail business acres per town
# 
# - CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# 
# - NOX nitric oxides concentration (parts per 10 million)
# 
# - RM average number of rooms per dwelling
# 
# - AGE proportion of owner-occupied units built prior to 1940
# 
# - DIS weighted distances to five Boston employment centres
# 
# - RAD index of accessibility to radial highways
# 
# - TAX full-value property-tax rate per $10,000
# 
# - PTRATIO pupil-teacher ratio by town
# 
# - B [1000*(Bk - 0.63) ^ 2] where Bk is the proportion of blacks by town
# 
# - LSTAT percentage lower status of the population
# 
# - MEDV Median value of owner-occupied homes in $1000s

# In[3]:


# add in column names
housing_colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing_df.columns = housing_colnames
housing_df.info()


# Based on the output above, the dataset doesn't contain any null values and consists of 506 samples and 14 features. Yay! We don't have to spend a bunch of time cleaning it up!

# **Exploring the data**:
# 
# I'm going to explore and visualize the data it to get a better idea of:
# 1. The relationship of the median value of house prices with other features in the dataset
# 2. The distribution characteristics of features in the dataset
# 3. How the features of the data are correlated with one another

# In[27]:


# import visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
# set seaborn style
sns.set_style("dark")
sns.set_style("ticks")
print()


# In[28]:


# defining a function to plot each continuous feature against the target variable to see if there are any obvious
# trends in the data. 
def plot_features(col_list, title):
    plt.figure(figsize=(10, 14));
    i = 0
    for col in col_list:
        i += 1
        plt.subplot(6,2,i)
        plt.plot(housing_df[col], housing_df['MEDV'], marker='.', linestyle='none')
        plt.title(title % (col))
        plt.tight_layout()


# In[29]:


plot_colnames = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
plot_features(plot_colnames, "Relationship of %s vs Median House Value")


# The figure above provides a nice graphical summary of the relationships between median house value and the other features in the dataset. Average number of rooms per dwelling (RM) and Median House Value (MEDV) seem to have a reasonably linear relationship when you eyeball it (we'll look at whether the relationship between these two variables is actually linear later on in the notebook). 

# Another option for visualization is to use seaborns pairplot function, although with so many features in the dataset it can be difficult to read! Check it out below.

# In[30]:


print()
print()


# From a combination of looking at the initial scatter plots that I drew with Median House Value as a function of different features, and the pairplot above, we can see that there seems to be a linear relationship between Number of Rooms per Dwelling (RM) and Median House Value (MEDV). MEDV also appears to have a normal distribution, with a smattering of outliers (pair plot [14, 14]).

#  In the exploratory phase, a correlation matrix is also useful. It can help you to select features based on their linear correlation with the target variable, as well as quantifying their relationships with one another.

# In[31]:


corr_matrix = np.corrcoef(housing_df[housing_colnames].values.T)
# to control figure size
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set(font_scale=1.5)
# plot heatmap of correlations
print()
print()


# To select a good feature for a univariate linear regression, we want to select a variable that is strongly linearly correlated with our target variable, median value. Generally, variables with correlation coefficients above .7 are considered to be strongly correlated. From the matrix above we can see that this is the case for median value (MEDV) and rooms per dwelling(RM), which are positively correlated (0.7), and for MEDV and percentage lower status of the population (LSTAT) which are negatively correlated (-0.74).
# 
# Even though the correlation between LSTAT and MEDV is higher, the plots we made earlier depict a non-linear relationship between these two variables (see larger version below). 

# In[32]:


plt.plot(housing_df['LSTAT'], housing_df['MEDV'], marker='.', linestyle='none')
plt.xlabel('Percentage lower status of the population')
plt.ylabel('MEDV (1000s)')
plt.title('Median house value (MEDV) vs percentage lower status of the population')
print()


# On the other hand, the relationship between MEDV and RM does appear to be linear. Given this, RM and MEDV seem like good choices to model with a linear regression (see a bigger scatter plot of these variables below)

# In[11]:


plt.plot(housing_df['RM'], housing_df['MEDV'], marker='.', linestyle='none')
plt.xlabel('Average number of rooms per dwelling')
plt.ylabel('MEDV (1000s)')
plt.title('Median house value (MEDV) vs Average rooms per dwelling')
print()


# **Solving for regression parameters with gradient descent**
# 
# Based on the exploratory analysis above, I'll proceed with a univariate regression using MEDV as the target variable and RM (average rooms per dwelling) as the predictor variable. Because we are fiting a linear model, the hypothesis function for the problem looks like:

# In[34]:


Image("../../../input/apratim87_housingdata/math-images-for-linear-reg/image001.png")


# In[14]:


# isolating target and predictor variables
target = housing_df['MEDV']
predictor = housing_df['RM']


# As outlined above, we are going to model the relationship between Median House Value and Number of Rooms per Dwelling with a linear model with two parameters: 
# 

# In[35]:


Image("../../../input/apratim87_housingdata/math-images-for-linear-reg/image002.png")


# In order to choose what the values of these parameters should be, we have to decide what constitutes a 'good fit' to the training data that we have. We want to try to choose values for each parameter that generates a prediction of the house value when you input the number of rooms into the hypothesis function that is as close as possible to the values that we have in the training data. Mathematically, we want to solve the equation below (where m is the number of samples in the training set).

# In[39]:


Image("../../../input/apratim87_housingdata/math-images-for-linear-reg/image004.png")


# The above equation can be defined as a cost function AKA squared error cost function i.e.

# In[37]:


Image("../../../input/apratim87_housingdata/math-images-for-linear-reg/image005.png")


# In[38]:


Image("../../../input/apratim87_housingdata/math-images-for-linear-reg/image006.png")


# Now I can implement the cost function for a linear regression using the mean squared error. The mean squared error is the sum of the 'distance' or error between the actual points in the training set, and what the hypothesis function predicted it would be.

# In[40]:


def compute_cost(X, y, theta):
    return np.sum(np.square(np.matmul(X, theta) - y)) / (2 * len(y))

theta = np.zeros(2)
# adds a stack of ones to vectorize the cost function and make it useful for multiple linear regression 
X = np.column_stack((np.ones(len(predictor)), predictor))
y = target
cost = compute_cost(X, y, theta)

print('theta:', theta)
print('cost:', cost)


# When both parameters are set to 0 the mean squared error is ~296. We can now minimize the cost function using a method called gradient descent. Gradient descent minimizes the cost function by taking small steps down the slope of the cost function in each feature dimension. The size of each step down the function is determined by the partial derivative of the cost function with respect to the feature and a learning rate multiplier (alpha). If these two determinants are tuned appropriately,  gradient descent will successfully converge on a global minimum by iteratively adjusting feature weights (theta) of the cost function.

# In[41]:


def gradient_descent(X, y, alpha, iterations):
    """function that performs gradient descent"""
    theta = np.zeros(2)
    m = len(y)
    iteration_array = list(range(1, iterations + 1))
    cost_iter = np.zeros(iterations)
    for i in range(iterations):
        t0 = theta[0] - (alpha / m) * np.sum(np.dot(X, theta) - y)
        t1 = theta[1] - (alpha / m) * np.sum((np.dot(X, theta) - y) * X[:,1])
        theta = np.array([t0, t1])
        cost_iter[i] = compute_cost(X, y, theta)
    cost_iter = np.column_stack((iteration_array, cost_iter))
    return theta, cost_iter


# In[42]:


iterations = 1000
alpha = 0.01

theta, cost_iter = gradient_descent(X, y, alpha, iterations)
cost = compute_cost(X, y, theta)

print("theta:", theta)
print('cost:', compute_cost(X, y, theta))


# It can be helpful to plot the cost as gradient descent runs through each iteration to check what point / if it converges successfully. In the plot below it looks as though gradient descent didn't coverge until after 6000 iterations. You can also see the effects of playing with the alpha value on gradient descent - if alpha is set too small, it takes more iterations to converge. If its set too large then you run the risk of it not converging at all and the cost becoming infinitely large.

# In[43]:


plt.plot(cost_iter[:,0], cost_iter[:,1])
plt.xlabel('Gradient descent iteration #')
plt.ylabel('Cost')
print()


# Alright, lets produce a scatter plot with the data and the regression line that results from gradient descent.

# In[44]:


plt.scatter(predictor, target, marker='.', color='green')
plt.xlabel('Number of Rooms per Dwelling')
plt.ylabel('Median House Value in 1000s')
samples = np.linspace(min(X[:,1]), max(X[:,1]))
plt.plot(samples, theta[0] + theta[1] * samples, color='black')


# In the graph above, you can see that the regression has successfully fit a line to the dataset, although in many cases the model wouldn't predict house value very accurately. There is also a strange set of datapoints with a house value of $50,000, irrespective of the number of rooms in the house.

# A surface plot provides a better illustration of how gradient descent determines a global minimum. The surface plot below plots the values for θ against their cost.

# In[45]:


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

Xs, Ys = np.meshgrid(np.linspace(-35, 0, 50), np.linspace(0, 15, 50))
Zs = np.array([compute_cost(X, y, [t0, t1]) for t0, t1 in zip(np.ravel(Xs), np.ravel(Ys))])
Zs = np.reshape(Zs, Xs.shape)

fig = plt.figure(figsize=(8,8))
ax = fig.gca(projection="3d")
ax.set_xlabel(r'theta 0 ', labelpad=20)
ax.set_ylabel(r'theta 1 ', labelpad=20)
ax.set_zlabel(r'cost  ', labelpad=10)
ax.view_init(elev=25, azim=40)
ax.plot_surface(Xs, Ys, Zs, cmap=cm.rainbow)


# Another way to visualize the cost function is using a contour plot, which depicts slices of the surface plot in a 2D space. In the contour plot below the red star shows the θ values sitting at the global minimum.

# In[46]:


ax = plt.figure().gca()
ax.plot(theta[0], theta[1], 'r*')
plt.contour(Xs, Ys, Zs, np.logspace(-3, 3, 15))
ax.set_xlabel(r'theta 0')
ax.set_ylabel(r'theta 1')


# **Univariate linear regression in sklearn**
# 
# Luckily, python has some great libraries that make estimating the regression parameters really easy. One of these is [sklearn](http://scikit-learn.org/stable/index.html). Below I show how to fit a simple linear regression using sklearns LinearRegression function from their linear model library.

# In[47]:


# import packages
from sklearn import preprocessing
from sklearn import linear_model

# data
# load data
housing_df = pd.read_csv("../../../input/apratim87_housingdata/housingdata.csv", header=None)
# add in column names
housing_colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing_df.columns = housing_colnames
# select target and feature

X = housing_df[['RM']].values
# MEDV
y = housing_df['MEDV']
# instantiate the linear model
linear = linear_model.LinearRegression()
# fit the model
linear.fit(X,y)
# generate plot
plt.scatter(X, y, c='blue')
plt.plot(X, linear.predict(X), marker='.',color='red')
plt.xlabel("Number of rooms")
plt.ylabel("House value in 1000's")
print()

# print fit info
print ('='* 65)
print ('%30s: %s' % ('Model R-squared', linear.score(X, y)))
print ('%30s: %s' % ('Slope', linear.coef_[0]))
print ('%30s: %s' % ('Model intercept', linear.intercept_))
print ('='* 65)


# You can see that the parameter estimates output by gradient descent differ from the ones determined using sklearns LinearRegression. 

# **Robust univariate linear regression with sklearn's RANSAC algorithm**
# A second option from sklearn is using its RANSAC algorithm to fit a linear model while dealing with outliers systematically.
# The RANSAC algorithm provides an alternative to removing outliers. Instead of relying on the investigator to pick and remove outliers, it fits a regression model to a subset of the data (referred to as inliers). The steps in the RANSAC are:
# 1. Select inliers using random sampling
# 2. Test all the other data points against the fitted model and add the points that fall within a user-specified tolerance to the inliers
# 3. Refit using all the inliers
# 4. Estimate the error of the fitted model versus the inliers
# 5. Terminate the algorithm if the performance meets a user-specified threshold or if a fixed number of iterations is performed.

# In[48]:


from sklearn.linear_model import RANSACRegressor

# load data
housing_df = pd.read_csv("../../../input/apratim87_housingdata/housingdata.csv", header=None)
# add in column names
housing_colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing_df.columns = housing_colnames
# select target and feature

X = housing_df[['RM']].values
# MEDV
y = housing_df['MEDV']

# instantiate the linear model
linear = linear_model.LinearRegression()

ransac = RANSACRegressor(linear, max_trials=100, min_samples=50, residual_threshold=5.0, random_state=0) 
ransac.fit(X,y)


# In[49]:


# plot inliers and outliers determined by RANSAC
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='.', label='Inliers') 
plt.scatter(X[outlier_mask], y[outlier_mask], c='green', marker='.', label='Outliers') 
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('Rooms per Dwelling')
plt.ylabel('House value in $1000s')
plt.legend(loc='lower right')
print()

# print fit info
print ('='* 65)
print ('%30s: %s' % ('Model R-squared', ransac.estimator_.score(X, y)))
print ('%30s: %s' % ('Slope', ransac.estimator_.coef_[0]))
print ('%30s: %s' % ('Model intercept', ransac.estimator_.intercept_))
print ('='* 65)


# **Performance evaluation of linear regression models**
# 
# The section above goesthrough how to fit a regression model on some dataset that you have. In order to assess its performance i.e. how 'good' it is at predicting house prices when given the number of rooms in the dwelling, we should really train the model on a portion of the full dataset (aka the 'training' set) and use the remaining proportion of the dataset (aka the 'test' set) to get an estimate of the performance of the model.

# In[50]:


from sklearn.cross_validation import train_test_split

X = housing_df[['RM']].values
y = housing_df[['MEDV']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
y_train_pred= lr.predict(X_train)
y_test_pred = lr.predict(X_test)


# In[51]:


# plot a residual plot consisting of points where the true target values are subtracted from the predicted responses
_ = plt.scatter(y_train_pred, y_train_pred - y_train, c='red', marker='o', label='Training data')
_ = plt.scatter(y_test_pred, y_test_pred - y_test, c='green', marker='o', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='blue')
plt.xlim([-10, 50])
print()


# In[52]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('MSE train:', (mean_squared_error(y_train, y_train_pred)))
print('MSE test:', (mean_squared_error(y_test, y_test_pred)))
print('r-sqaured train:', (r2_score(y_train, y_train_pred)))
print('r-squared test:', (r2_score(y_test, y_test_pred)))   


# **Multiple linear regression**
# 
# Now I'm to perform a linear regression with multiple predictor variables to estimate house value using gradient descent .
# Based on the initial plots, it looks as though the relationships between house value and weighted distances to five Boston employment centres (DIS), house value and nitrous oxide concentration (NOX) , and house value and proportion of non-retail business acres per town (INDUS) could be modeled with a linear relationship. Lets add these to the predictor training set.

# In[57]:


# load data
housing_df = pd.read_csv("../../../input/apratim87_housingdata/housingdata.csv", header=None)
# add in column names
housing_colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing_df.columns = housing_colnames

predictors = housing_df.as_matrix(columns=['RM', 'DIS', 'NOX', 'INDUS'])
target = housing_df['MEDV']


# All of the predictors are measured on a different scale, which can cause issues with gradient descent. Having features on wildly different scales can lead to certain features disproportionately affecting the results. As such, it is common to scale the features using some normalization method e.g. min-max scaling or Z-score standardization. Here, we'll use Z-score standardization to scale the features.

# In[58]:


Image("../../../input/apratim87_housingdata/math-images-for-linear-reg/image007.png")


# where mean:

# In[59]:


Image("../../../input/apratim87_housingdata/math-images-for-linear-reg/image008.png")


# and standard deviation:
# 

# In[60]:


Image("../../../input/apratim87_housingdata/math-images-for-linear-reg/image009.png")


# In[61]:


def znormalize_features(X):
    n_features = X.shape[1]
    means = np.array([np.mean(X[:,i]) for i in range(n_features)])
    stddevs = np.array([np.std(X[:,i]) for i in range(n_features)])
    normalized = (X - means) / stddevs

    return normalized

X = znormalize_features(predictors)
X = np.column_stack((np.ones(len(X)), X))
# scale target variable
y = target / 1000


# In[62]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Before Scaling')
sns.kdeplot(predictors[:,0], ax=ax1)
sns.kdeplot(predictors[:,1], ax=ax1)
sns.kdeplot(predictors[:,2], ax=ax1)
sns.kdeplot(predictors[:,3], ax=ax1)
ax2.set_title('After Scaling')
sns.kdeplot(X[:,1], ax=ax2)
sns.kdeplot(X[:,2], ax=ax2)
sns.kdeplot(X[:,3], ax=ax2)
sns.kdeplot(X[:,4], ax=ax2)
print()


# Next, we need to generalize the single feature case that we did earlier so that we can perform gradient descent with multiple features.

# In[63]:


def multi_variate_gradient_descent(X, y, theta, alpha, iterations):
    theta = np.zeros(X.shape[1])
    m = len(X)

    for i in range(iterations):
        gradient = (1/m) * np.matmul(X.T, np.matmul(X, theta) - y)
        theta = theta - alpha * gradient

    return theta


# In[64]:


theta = multi_variate_gradient_descent(X, y, theta, alpha, iterations)
cost = compute_cost(X, y, theta)

print('theta:', theta)
print('cost', cost)


# We can check that gradient descent acheived a reasonable answer by using the normal equation, which can solve for theta without needing an alpha value or iteration:

# In[65]:


Image("../../../input/apratim87_housingdata/math-images-for-linear-reg/image010.png")


# Note: using the normal equation is more computationally effective than gradient descent, and performs well up to about 1000 features. Beyond 1000 features gradient descent performs better.

# In[66]:


from numpy.linalg import inv

def normal_eq(X, y):
    return inv(X.T.dot(X)).dot(X.T).dot(y)

theta = normal_eq(X, y)
cost = compute_cost(X, y, theta)

print('theta:', theta)
print('cost:', cost)

