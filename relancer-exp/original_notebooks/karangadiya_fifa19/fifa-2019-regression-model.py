#!/usr/bin/env python
# coding: utf-8

# OBJECTIVE : The dataset contains detailed attributes for every player registered in the latest edition of FIFA 19 database. Our objective is to create Linear, Multiple and Polynomail Regression models to predict the potential of a player based on several attributes.

# In[88]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# In[89]:


# reading dataset 
data=pd.read_csv("../../../input/karangadiya_fifa19/data.csv")


# In[90]:


# displaying first 5 rows
data.head()


# In[91]:


data.shape #(no. of rows, no. of columns)


# In[92]:


data.describe()


# In[93]:


# finding any null values in data
data.isnull().any()


# #  Linear Regression - Predicting Potential based on Age of the player

# In[94]:


# x = Age(independent variable)
x=data.iloc[:,3] 


# In[95]:


x.head()


# In[96]:


x.isnull().any()


# In[97]:


# y = Potential(dependent variable)
y=data.iloc[:,8]


# In[98]:


y.head()


# In[99]:


y.isnull().any()


# In[100]:


plt.bar(data["Age"],data["Potential"])
plt.xlabel("Age of Player")
print()


# In[101]:


# splitting data into train and tet set
from sklearn.model_selection import train_test_split


# In[102]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[103]:


from sklearn.linear_model import LinearRegression


# In[104]:


# making object regressor of class LinearRegression
regressor=LinearRegression()


# I was facing errors with fitting the data so we reshape x_train and y_train by first converting them into ndarray.

# In[105]:


type(x_train)
type(y_train)


# In[106]:


x_train=np.array(x_train)
y_train=np.array(y_train)


# In[107]:


type(x_train)
type(y_train)


# In[108]:


x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)


# In[109]:


# fitting training set into object regressor
regressor.fit(x_train,y_train)


# To avoid error in prediting we reshape x_test also.

# In[110]:


x_test=np.array(x_test)


# In[111]:


x_test=x_test.reshape(-1,1)


# In[112]:


# Predicting y from test set
y_pred= regressor.predict(x_test)


# In[113]:


# Visualising training dataset
plt.scatter(x_train,y_train,color="red")
plt.xlabel("Age of Player")
plt.ylabel("Potential of Player")
plt.plot(x_train, regressor.predict(x_train),color="blue") # To draw line of regression
print()


# In[114]:


# Visualising test dataset
plt.scatter(x_test,y_test,color="red")
plt.xlabel("Age of Player")
plt.ylabel("Potential of Player")
plt.plot(x_train, regressor.predict(x_train),color="blue")
print()


# In[115]:


# Finding intercept of linear regression line
regressor.intercept_


# In[116]:


# Finding coefficient of linear regression line
regressor.coef_


# In[117]:


# Finding mean squared error of linear regression model
from sklearn.metrics import mean_squared_error


# In[118]:


mean_squared_error(y_test,y_pred)


# # Multiple regression - Predicting potential based on age, agility, balance, stamina, strength, composure
# 

# In[119]:


# independent variables are - Age, Agility, Balance, stamina, Strength, Composure
x=data.iloc[:,[3,66,68,71,72,79]]


# In[120]:


x.head()


# In[121]:


# checking if there are null values in x and then filling them. 
x.isnull().any()


# In[122]:


x=x.fillna(method='ffill')


# In[123]:


x.isnull().any()


# In[124]:


# dependent variable = Potential
y=data.iloc[:,8]


# In[125]:


y.head()


# In[126]:


y.isnull().any()


# In[127]:


sns.lineplot(x="Potential", y="Age",data=data,label="Age", ci= None)
sns.lineplot(x="Potential", y="Agility",data=data,label="Agility", ci= None)
sns.lineplot(x="Potential", y="Balance",data=data,label="Balance", ci= None)
sns.lineplot(x="Potential", y="Stamina",data=data,label="Stamina", ci= None)
sns.lineplot(x="Potential", y="Strength",data=data,label="Strength", ci= None)
sns.lineplot(x="Potential", y="Composure",data=data,label="Composure", ci= None)


# In[128]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[129]:


regressor=LinearRegression()


# In[130]:


regressor.fit(x_train,y_train)


# In[131]:


regressor.predict(x_test)


# In[132]:


# Visualising Actual and predicted values of Potential of player
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Potential")
plt.ylabel("Predicted Potential")
print()


# Seems like the actual and predicted values are very close to each other.

# In[133]:


regressor.intercept_


# In[134]:


regressor.coef_


# Backward Elimination - Making optimal regression model by finding the statistical significance of all independent variables

# In[135]:


# let us take the significance level (SL)= 0.05
import statsmodels.formula.api as sm


# In[136]:


# fitting all variables in the model
regressor_OLS=sm.OLS(endog=y,exog=x).fit()


# In[137]:


# Finding statistical summary of all variables
regressor_OLS.summary()


# As we see all the P- values are less than SL(0.05), that means all the variables are significant and none of them can be removed. 
# t-value shows the statistical significane of each variable.
# F-static shows us how significant the fit is. 
# Adjusted- R is 0.986 that means our model explains 98.6% variables in dependent variables.

# # Polynomial Regression - Predicting potential based on the age of player

# In[138]:


# independent variable= age
x=data.iloc[:,3]


# In[139]:


x.head()


# In[140]:


# dependent variable = potential
y=data.iloc[:,8]


# In[141]:


y.head()


# In[142]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[143]:


x_train=np.array(x_train)
y_train=np.array(y_train)


# In[144]:


x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)


# In[145]:


lin_reg_1=LinearRegression()


# In[146]:


lin_reg_1.fit(x_train,y_train)


# In[147]:


x_test=np.array(x_test)


# In[148]:


x_test=x_test.reshape(-1,1)


# In[149]:


y_pred_1=lin_reg_1.predict(x_test)


# In[150]:


# Making polynomail regression model
from sklearn.preprocessing import PolynomialFeatures


# In[151]:


poly_reg=PolynomialFeatures(degree=3)


# In[152]:


x=np.array(x)


# In[153]:


x=x.reshape(-1,1)


# In[154]:


# Making polynomial matrix of x of degree 3
x_poly=poly_reg.fit_transform(x)


# In[155]:


x_poly


# In[156]:


x_poly_train,x_poly_test,y_train,y_test=train_test_split(x_poly,y,test_size=0.2, random_state=42)


# In[157]:


# Making another object to fit polynomial set
lin_reg_2=LinearRegression()


# In[158]:


lin_reg_2.fit(x_poly_train,y_train)


# In[159]:


y_pred_2=lin_reg_2.predict(x_poly_test)


# In[160]:


# Visualizing Linear Regression Model
plt.scatter(x_test,y_test,color='red')
plt.xlabel("Age of Player")
plt.ylabel("Potential of Player")
plt.title("Linear Regression Curve ")
plt.plot(x_train,lin_reg_1.predict(x_train),color='blue')
print()


# In[161]:


# Visualizing Polynomial Regression Model
plt.scatter(x_test,y_test,color='red')
plt.xlabel("Age of Player")
plt.ylabel("Potential of Player")
plt.title("Polynomial Regression Curve ")
plt.plot(x_train,lin_reg_2.predict(poly_reg.fit_transform(x_train)),color='blue')
print()


# In[162]:


mean_squared_error(y_test,y_pred_2)


# We can see the mean squared error of polynomail regression model < mean squared error of linear regression model. So polynomail regression model is more accurate.
