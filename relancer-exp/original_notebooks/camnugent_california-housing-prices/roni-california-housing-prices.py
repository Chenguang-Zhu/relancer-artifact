#!/usr/bin/env python
# coding: utf-8

# # Regression Project: The California Housing Prices Data Set

# ### Context
# This is the dataset used in the second chapter of Aurélien Géron's recent book *'Hands-On Machine learning with Scikit-Learn and TensorFlow'. (O'Reilly)*
# 
# It serves as an excellent introduction to implementing machine learning algorithms because it requires rudimentary (first, primary, original, prime, primitive) data cleaning, has an easily understandable list of variables and sits at an optimal size between being to toyish and too cumbersome.
# 
# The data contains information from the 1990 California census.
# 
# ### Source
# This dataset is a modified version of the California Housing dataset available from Luís Torgo's page (University of Porto).
# 
# This dataset appeared in a 1997 paper titled *Sparse Spatial Autoregressions* by Pace, R. Kelley and Ronald Barry, published in the *Statistics and Probability Letters* journal. **It contains one row per census block group**. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data **(a block group typically has a population of 600 to 3,000 people)**.
# 
# Note that the **block groups are called "districts"** in the Jupyter notebooks, simply because in some contexts the name "block group" was confusing.
# 
# ### Content
# The data aren't cleaned so there are some preprocessing steps that were 
# required.
# 
# The data file weighs about 1.35 MB and has 20,640 rows and 10 columns.
# The names of the columns are pretty self explanatory:
# 1. longitude: A measure of how far west a house is; a higher value is farther west
# 2. latitude: A measure of how far north a house is; a higher value is farther north
# 3. housing_median_age: Median age of a house within a block; a lower number is a newer building **("Block" == "District")**
# 4. total_rooms: Total number of rooms within a block
# 5. total_bedrooms: Total number of bedrooms within a block
# 6. population: Total number of people residing within a block
# 7. households: Total number of households, a group of people residing within a home unit, for a block
# 8. median_income: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
# 9. median_house_value: Median house value for households within a block (measured in US Dollars)
# 10. ocean_proximity: Location of the house w.r.t ocean/sea
# 
# ### Tweaks
# The dataset in this directory is almost identical to the original, with two differences:
# - 207 values were randomly removed from the **total_bedrooms** column, so we can 
#   discuss what to do with **missing data**.
# - An additional categorical attribute called **ocean_proximity** was added,#   the Bay area, inland or on an island. This allows discussing what to do with    **categorical data**.
# 
# 

# ## Import statements
# 
# 

# In[ ]:


# General tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# statsmodels package
import statsmodels.api as sm
import statsmodels.formula.api as smf

# For transformations and predictions
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from itertools import product
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold, cross_val_score
from sklearn.decomposition import PCA

# For the tree visualization
import pydot
from IPython.display import Image
from sklearn.externals.six import StringIO

# For scoring
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error as msle

# For split
from sklearn.model_selection import train_test_split as split

from sys import modules

import warnings
warnings.filterwarnings('ignore')


# ## Get the data

# **Upload**

# In[ ]:


if 'google.colab' in modules:
    from google.colab import files
    uploaded = files.upload()


# **Read file**

# In[ ]:


housing_df = pd.read_csv("../../../input/camnugent_california-housing-prices/housing.csv")
housing_df.shape


# In[ ]:


housing_df


# ## EDA with map visualization
# One good practice is to do EDA on the full data and **creating a copy** of it for not harming our test and training data.

# In[ ]:


plotter_df = housing_df.copy()


# In[ ]:


from PIL import Image
plt.figure(figsize=(20,10))
img = np.array(Image.open("california.jpg"))
plt.imshow(img, interpolation = "bilinear")
plt.axis("off")
print()


# **Since there is geographical information (latitude and longitude), it is a good idea to create a scatterplot of all districts to visualize the data.**

# In order to have an informative look on the plot we need to know the density for each point, so we use alpha=0.1 (transparency measure).

# In[ ]:


plt.figure(figsize=(10,8))
plt.scatter('longitude', 'latitude', data=plotter_df,alpha=0.1)
plt.ylabel('Latitudes')
plt.xlabel('Longitudes')
plt.title('Geographical plot of Latitudes/Longitudes')
print()


# **median_income** influence on **median_house_value.**

# In[ ]:


plotter_df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,s=plotter_df['median_income']*30, label='median_income', figsize=(10,7),c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.xlabel("Longitude", fontsize=14)
plt.ylabel("Latitude", fontsize=14)
plt.legend() 
print()


# ## Look at the Data

# In[ ]:


housing_df.head()


# In[ ]:


housing_df.describe(include='all')


# In[ ]:


housing_df.info()


# We can see that the 'total_bedrooms' column has only 20,433 non-null values compared to 20,640 non-null values in other columns.
# 
# 

# In[ ]:


housing_df.isna().sum()


# # Data Cleaning

# ## Handling missing values

# Most Machine Learning algorithms cannot work with missing features, so let’s create a few functions to take care of them. We noticed earlier that the total_bedrooms attribute has some missing values, so let’s fix this. We have three options:
#  1. Get rid of the corresponding districts.
#  2. Get rid of the whole attribute.
#  3. Set the values to some value (zero, the mean, the median, etc.)

# In[ ]:


# Option 1: dropping rows (districts) with missing values
housing_df.dropna(subset=["total_bedrooms"], inplace=True)
housing_df.shape


# In[ ]:


# Option 2: dropping the whole column
#housing_df.drop("total_bedrooms", axis=1, inplace=True)
#housing_df.shape


# In[ ]:


# Option 3: fill NA values
#median_total_bedrooms = housing_df["total_bedrooms"].median() 
#housing_df["total_bedrooms"].fillna(median_total_bedrooms, inplace=True)
#housing_df.shape


# ## Handling ocean_proximity categorical column.

# In[ ]:


housing_df['ocean_proximity'].value_counts()


# In[ ]:


housing_df.boxplot(column=['median_house_value'], by='ocean_proximity')


# We can see that there are only 5 rows with 'ocean_proximity'=='ISLAND'.
# We decided to drop these values.

# In[ ]:


housing_df.drop(housing_df[housing_df['ocean_proximity']=='ISLAND'].index, inplace=True)
housing_df.shape


# ### Transform categorical data to discrete values

# In[ ]:


ocean_proximity_order = ['INLAND','<1H OCEAN', 'NEAR OCEAN', 'NEAR BAY']
#ocean_proximity_order = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
ocean_proximity_map = dict(zip(ocean_proximity_order, range(len(ocean_proximity_order))))

housing_df['ocean_proximity'] = housing_df['ocean_proximity'].map(ocean_proximity_map)


# In[ ]:


housing_df['ocean_proximity'].value_counts()


# ## Creating new features and deleting unnecessary features‏

# 1. The total number of rooms in a district is not very useful if you don’t know how many
# households there are. What we really want is the number of rooms per household.
# 
# 2. Similarly, the total number of bedrooms by itself is not very useful: We  want to compare it to the number of rooms.
# 
# 3. The population per household also seems like an interesting attribute combination to look at.

# In[ ]:


housing_df["rooms_per_household"] = housing_df["total_rooms"]/housing_df["households"]
housing_df["bedrooms_per_room"] = housing_df["total_bedrooms"]/housing_df["total_rooms"]
housing_df["population_per_household"]=housing_df["population"]/housing_df["households"]


# In[ ]:


# dropping unnecessary features
housing_df.drop(["total_rooms","total_bedrooms","population"], axis=1, inplace=True)
housing_df.head()


# In[ ]:


housing_df.shape


# ## Split the data to train and test

# In[ ]:


housing_df_train, housing_df_test  = split(housing_df, test_size=0.3, random_state=43)


# In[ ]:


housing_df_train.shape


# ## Removing Outliers
# 
# we can see median_house_value and housing_median_age have "peaks"
# and this data should be removed.

# In[ ]:


housing_df_train.hist(bins=50,figsize=(20,15))


# In[ ]:


housing_df_train = housing_df_train[(housing_df_train.median_house_value < 500001) & (housing_df_train.housing_median_age < 52)]  
housing_df_train.shape


# Now we drop data which is not in the range of 3 sigma (+/-).

# In[ ]:


for col in housing_df_train.columns:
    if housing_df_train[col].dtype == 'float64':
        std = housing_df_train[col].std()
        ave = housing_df_train[col].mean()
        housing_df_train_1 = housing_df_train.loc[housing_df_train[col].between(ave-3*std, ave+3*std)]
        print(f'processing {col:10} --> {housing_df_train_1.shape[0]:5} rows remain')


# # Linear Regression model: Assumptions of Linear Regression:
# 
# 1) Linear Relationship between the features and target
# 
# 2) Little or no Multicollinearity between the features
# 
# 3) Homoscedasticity
# 
# 4) Normal distribution of error
# 
# 5) Little or No autocorrelation in the residuals

# ### Check skewness (a measure of the asymmetry of a probability distribution)

# In[ ]:


housing_df_train_1.skew()


# In[ ]:


#log_rooms_per_household = np.log1p(housing_df_train_1['rooms_per_household'])
#log_population_per_household = np.log1p(housing_df_train_1['population_per_household'])
#log_households = np.log1p(housing_df_train_1['households'])
#log_bedrooms_per_room = np.log1p(housing_df_train_1['bedrooms_per_room'])

#housing_df_train_1['log_rooms_per_household'] = log_rooms_per_household
#housing_df_train_1['log_population_per_household'] = log_population_per_household
#housing_df_train_1['log_households'] = log_households
#housing_df_train_1['log_bedrooms_per_room'] = log_bedrooms_per_room

# dropping unnecessary features
#housing_df_train_1.drop(["rooms_per_household","population_per_household","households","bedrooms_per_room" ], axis=1, inplace=True)

#housing_df_train_1.skew()


# In[ ]:


housing_df_train_1.head()


# ## Split and scaling the data and preparation to model

# In[ ]:


X_train = housing_df_train_1.drop(['median_house_value'],axis=1)
y_train = housing_df_train_1['median_house_value']
# scaling the data
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_train_scaled.head()



# #Linear Relationship between the features and target

# ## Correlation

# In[ ]:


corr_matrix = housing_df_train_1.corr()


# we can see that median_income and ocean_proximity has the most efect on 
# median_house_value

# In[ ]:


corr_matrix['median_house_value'].sort_values(ascending=False)


# **We can see a small negative correlation between the latitude and the  median house value (i.e., prices have a slight tendency to go down when you go north.**

# # Assumption check: Little or no Multicollinearity between the features

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = X_train_scaled
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns


# In[ ]:


vif


# In[ ]:


X_train_scaled_1 = X_train_scaled.drop(['longitude','bedrooms_per_room'],axis=1)


# fit the linear model

# In[ ]:


regr = LinearRegression()
regr.fit(X_train_scaled_1, y_train)
y_train_pred = regr.predict(X_train_scaled_1)
resids = y_train - y_train_pred


# In[ ]:


RMSE = mse(y_train, y_train_pred)**0.5
print('RMSE:',RMSE)


# 
# 
# #Homoscedasticity

# In[ ]:


sns.set_style('darkgrid')
sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)
fig, ax = plt.subplots(1,2)
    
sns.regplot(x=y_train_pred, y=y_train, lowess=True, ax=ax[0], line_kws={'color': 'red'})
ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
ax[0].set(xlabel='Predicted', ylabel='Observed')

sns.regplot(x=y_train_pred, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})
ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
ax[1].set(xlabel='Predicted', ylabel='Residuals')
      


#  **Validate the model** 

# In[ ]:


X_test = housing_df_test.drop(['median_house_value', 'longitude','bedrooms_per_room'],axis=1)
y_test = housing_df_test['median_house_value']


# In[ ]:


scaler1 = MinMaxScaler()
scaler1.fit(X_test)

X_test_scaled = pd.DataFrame(scaler1.transform(X_test), columns=X_test.columns)
X_test_scaled.head()


# In[ ]:


y_test_pred = regr.predict(X_test_scaled)
resids_test = y_test - y_test_pred


# In[ ]:


RMSE = mse(y_test, y_test_pred)**0.5
print('RMSE:',RMSE)


# **Important conclusion**
# 
# The model is valid! It is not useful, but it is valid.
# 
# There was nothing in the process that could have indicated that the model is wrong. This is a fact of life - valid models are not necessarily useful.

# # Decision Tree 

# ## Tests for finding the most influential columns

# In[ ]:


dt_model_test = DecisionTreeRegressor(max_depth=8, min_samples_leaf=50, min_impurity_decrease=0.005)
#dt_model_test = DecisionTreeRegressor(min_samples_split=0.01, min_impurity_split=0.01)


# In[ ]:


dt_model_test.fit(X_train, y_train)


# In[ ]:


X.columns


# In[ ]:


for feature, importance in zip(X_train.columns, dt_model_test.feature_importances_):
    print('{:25}: {}'.format(feature, importance))


# In[ ]:


X_test_1 = housing_df_test.drop(['median_house_value'],axis=1)


# ## Prediction 

# In[ ]:


#Leaving the feature importances
X_train_dt = X_train.drop(['housing_median_age', 'households','rooms_per_household','bedrooms_per_room'],axis=1)

#check for data set
X_test_dt = X_test_1.drop(['housing_median_age', 'households','rooms_per_household', 'bedrooms_per_room'],axis=1)


# Loop with different hyper-parameters values

# In[ ]:


l_max_depth = [10, 15, 20, 30 ]
l_min_samples_leaf = [ 8, 50, 100, 150]
l_min_impurity_decrease = [0.01, 0.1]


# In[ ]:


from itertools import product


# In[ ]:


scores = []
for i, (md, msl, mid) in enumerate(product(l_max_depth, l_min_samples_leaf, l_min_impurity_decrease)):
    dt_model = DecisionTreeRegressor(max_depth=md, min_samples_leaf=msl, min_impurity_decrease=mid)
    dt_model.fit(X_train_dt, y_train)
    y_train_pred = dt_model.predict(X_train_dt)
    RMSE_train = mse(y_train, y_train_pred)**0.5
    y_test_pred = dt_model.predict(X_test_dt)
    RMSE_test = mse(y_test, y_test_pred)**0.5
    scores.append((md, msl, mid, RMSE_train, RMSE_test))


# In[ ]:


sorted(scores, key=lambda x: x[3], reverse=False)


# ## One of the better example for max_depth=30, min_samples_leaf=50, min_impurity_decrease=0.01

# In[ ]:


dt_model = DecisionTreeRegressor(max_depth=30, min_samples_leaf=50, min_impurity_decrease=0.01)
dt_model.fit(X_train_dt, y_train)
y_train_pred = dt_model.predict(X_train_dt)


# In[ ]:


RMSE_train = mse(y_train, y_train_pred)**0.5
RMSE_train


# In[ ]:


ax = sns.scatterplot(x=y_train, y=y_train_pred)
ax.plot(y_train, y_train, 'r')


# ## Validating the model

# In[ ]:


y_test_pred = dt_model.predict(X_test_dt)


# In[ ]:


RMSE_test = mse(y_test, y_test_pred)**0.5
RMSE_test


# In[ ]:


ax = sns.scatterplot(x=y_test, y=y_test_pred)
ax.plot(y_test, y_test, 'r')


# # K-Nearest Neighbors

# ## Loop with different n_neighbors values

# In[ ]:


X_train_knn = X_train_dt
X_test_knn = X_test_dt


# In[ ]:


scores = [] # to store rmse values for different k
for k in range(20):
    k = k+1
    knn_model = KNeighborsRegressor(n_neighbors = k).fit(X_train_knn, y_train)
    y_train_pred = knn_model.predict(X_train_knn)
    RMSE_train = mse(y_train, y_train_pred)**0.5
    y_test_pred = knn_model.predict(X_test_knn)
    RMSE_test = mse(y_test, y_test_pred)**0.5
    scores.append((k, RMSE_train, RMSE_test))
    
    


# In[ ]:


sorted(scores, key=lambda x: x[1], reverse=False)


# ## One of the better example for k=10 

#  Training the model for k=10

# In[ ]:


knn_model = KNeighborsRegressor(n_neighbors=10).fit(X_train_knn, y_train)
y_train_pred = knn_model.predict(X_train_knn)


# In[ ]:


RMSE_train = mse(y_train, y_train_pred)**0.5
RMSE_train


# In[ ]:


ax = sns.scatterplot(x=y_train, y=y_train_pred)
ax.plot(y_train, y_train, 'r')


#  Validating the model for k=10

# In[ ]:


y_test_pred = knn_model.predict(X_test_dt)


# In[ ]:


RMSE = mse(y_test, y_test_pred)**0.5
RMSE


# In[ ]:


ax = sns.scatterplot(x=y_test, y=y_test_pred)
ax.plot(y_test, y_test, 'r')


# # Conclusions:
# 
# ### The features that impact the most the median_house_value in a district in California in 1990 were:
# 
# - median_income.
# 
# - ocean proximity: the closer is a house to the ocean, the more it costs.
# 
# - latitude: we can see a small negative correlation between the latitude and the median house value (i.e., prices have a slight tendency to go down when you go north.

