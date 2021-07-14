#!/usr/bin/env python
# coding: utf-8

# # DIAMONDS IN-DEPTH ANALYSIS
# * You can also view the notebook on the link below.
# * *Github Link* - **https://github.com/Chinmayrane16/Diamonds-In-Depth-Analysis**
# * **Do Upvote if you like it :)**

# # Topics
# 1. [**Exploring Dataset**](#there_you_go_1)
# 2. [**Correlation b/w Features**](#there_you_go_2)
# 3. [**Visualizations**](#there_you_go_3)
# 4. [**Feature Engineering**](#there_you_go_4)
# 5. [**Feature Encoding**](#there_you_go_5)
# 6. [**Feature Scaling**](#there_you_go_6)
# 7. [**Modelling Algorithms**](#there_you_go_7)
# 8. [**Comparing R2 Scores**](#there_you_go_8)

# ## What are Diamonds?
# * **Diamonds are the Precious stone consisting of a clear and colourless Crystalline form of pure carbon.**
# * **They are the hardest Gemstones known to man and can be scratched only by other Diamonds.**

# ![](https://i.imgur.com/blhMqmD.jpg)

# ## How Diamonds are formed?
# * **Diamonds are formed deep within the Earth about 100 miles or so below the surface in the upper mantle.**
# * **Obviously in that part of the Earth it’s very hot.** 
# * **There’s a lot of pressure, the weight of the overlying rock bearing down, so that combination of high temperature and high pressure is what’s necessary to grow diamond crystals in the Earth.**

# ## Why are Diamonds so Valuable?
# * **Whether it is a Rare book, a fine bottle of Scotch, or a Diamond, something that is Rare and Unique is often expensive.**
# * **But what makes it truly Valuable is that this Rarity coincides with the desire of many to possess it. ;)**
# * **Diamonds are Rare because of the Incredibly powerful forces needed to create them.**
# 
# 
# * **And therefore Diamonds are considered to be Very Costly.**

# <a id="there_you_go_1"></a>
# # 1) Explore Dataset & Examine what Features affect the Price of Diamonds.

# ## 1.1) Importing Libraries

# In[ ]:


# Ignore warnings :
import warnings
warnings.filterwarnings('ignore')


# Handle table-like data and matrices :
import numpy as np
import pandas as pd
import math 



# Modelling Algorithms :

# Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis

# Regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor




# Modelling Helpers :
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV , KFold , cross_val_score



#preprocessing :
from sklearn.preprocessing import MinMaxScaler , StandardScaler, Imputer, LabelEncoder



#evaluation metrics :

# Regression
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 

# Classification
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  



# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import missingno as msno



# Configure visualisations
mpl.style.use( 'ggplot' )
plt.style.use('fivethirtyeight')
sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)
params = { 'axes.labelsize': "large", 'xtick.labelsize': 'x-large', 'legend.fontsize': 20, 'figure.dpi': 150, 'figure.figsize': [25, 7] } 
plt.rcParams.update(params)


# In[ ]:


# Center all plots
from IPython.core.display import HTML


# ## 1.2) Extract Dataset
# * Specify the location to the Dataset and Import them.

# In[ ]:


df = pd.read_csv("../../../input/shivam2503_diamonds/diamonds.csv")
diamonds = df.copy()


# In[ ]:


# How the data looks
df.head()


# ## 1.3) Features
# * **Carat : ** Carat weight of the Diamond.
# * **Cut : ** Describe cut quality of the diamond.
# > * Quality in increasing order Fair, Good, Very Good, Premium, Ideal .
# * **Color : ** Color of the Diamond.
# > * With D being the best and J the worst.
# * **Clarity : ** Diamond Clarity refers to the absence of the Inclusions and Blemishes.
# > * (In order from Best to Worst, FL = flawless, I3= level 3 inclusions) FL, IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1, I2, I3
# * **Depth : ** The Height of a Diamond, measured from the Culet to the table, divided by its average Girdle Diameter.
# * **Table : ** The Width of the Diamond's Table expressed as a Percentage of its Average Diameter.
# * **Price : ** the Price of the Diamond.
# * **X : ** Length of the Diamond in mm.
# * **Y : ** Width of the Diamond in mm.
# * **Z : ** Height of the Diamond in mm.
# 
# *Qualitative Features (Categorical) : Cut, Color, Clarity. *
# 
# *Quantitative Features (Numerical) : Carat, Depth , Table , Price , X , Y, Z.*
# 
# 
# ### Price is the Target Variable.

# ![](https://i.imgur.com/Bbf0GWk.jpg)

# In[ ]:


# We'll Explore All the features in the Later Part, Now let's look for Null Values if any..


# ## 1.4) Drop the 'Unnamed: 0' column as we already have Index.

# In[ ]:


df.drop(['Unnamed: 0'] , axis=1 , inplace=True)
df.head()


# In[ ]:


df.shape


# In[ ]:


# So, We have 53,940 rows and 10 columns


# In[ ]:


df.info()


# ## 1.5) Examine NaN Values

# In[ ]:


# It seems there are no Null Values.
# Let's Confirm
df.isnull().sum()


# In[ ]:


msno.matrix(df) # just to visualize. no missing values.


# ### Great, So there are no NaN values.

# In[ ]:


df.describe()


# ### Wait
# * **Do you see the Min. Values of X, Y and Z. It can't be possible..!!**
# * **It doesn't make any sense to have either of Length or Width or Height to be zero..**

# ### Let's Have a look at them.

# In[ ]:


df.loc[(df['x']==0) | (df['y']==0) | (df['z']==0)]


# In[ ]:


len(df[(df['x']==0) | (df['y']==0) | (df['z']==0)])


# ### We can see there are 20 rows with Dimensions 'Zero'.
# * **We'll Drop them as it seems better choice instead of filling them with any of Mean or Median**

# ## 1.6) Dropping Rows with Dimensions 'Zero'.

# In[ ]:


df = df[(df[['x','y','z']] != 0).all(axis=1)]


# In[ ]:


# Just to Confirm
df.loc[(df['x']==0) | (df['y']==0) | (df['z']==0)]


# In[ ]:


# Nice and Clean. :)


# ## 1.7) Scaling of all Features

# In[ ]:


sns.factorplot(data=df , kind='box' , size=7, aspect=2.5)


# **The Values are Distributed over a Small Scale.**

# In[ ]:





# <a id="there_you_go_2"></a>
# # 2) Correlation Between Features

# In[ ]:


# Correlation Map
corr = df.corr()
print()


# ## CONCLUSIONS :
# **1. Depth is inversely related to Price.**
# > * This is because if a Diamond's Depth percentage is too large or small the Diamond will become '__Dark__' in appearance because it will no longer return an Attractive amount of light.
# 
# **2. The Price of the Diamond is highly correlated to Carat, and its Dimensions.**
# 
# **3. The Weight (Carat) of a diamond has the most significant impact on its Price. **
# > * Since, the larger a stone is, the Rarer it is, one 2 carat diamond will be more '__Expensive__' than the total cost of two 1 Carat Diamonds of the same Quality.
# 
# **4. The Length(x) , Width(y) and Height(z) seems to be higly related to Price and even each other.**
# 
# **5. Self Relation ie. of a feature to itself is 1 as expected.**
# 
# **6. Some other Inferences can also be drawn.**

# In[ ]:





# <a id="there_you_go_3"></a>
# # 3. Visualization Of All Features

# ## 3.1) Carat
# 
# * **Carat refers to the Weight of the Stone, not the Size.**
# * **The Weight of a Diamond has the most significant Impact on its Price.**
# * **Since the larger a Stone is, the Rarer it is, one 2 Carat Diamond will be more Expensive than the Total cost of two 1 Carat Diamonds of the Same Quality.**
# * **The carat of a Diamond is often very Important to People when shopping But it is a Mistake to Sacrifice too much quality for sheer size.**
# 
# 
# [Click Here to Learn More about How Carat Affects the Price of Diamonds.](https://www.diamondlighthouse.com/blog/2014/10/23/how-carat-weight-affects-diamond-price/)

# ![](https://i.imgur.com/hA3oat5.png)

# In[ ]:


# Visualize via kde plots


# In[ ]:


sns.kdeplot(df['carat'], shade=True , color='r')


# ### Carat vs Price

# In[ ]:


sns.jointplot(x='carat' , y='price' , data=df , size=5)


# ### It seems that Carat varies with Price Exponentially.

# In[ ]:





# ## 3.2) Cut
# 
# * **Although the Carat Weight of a Diamond has the Strongest Effect on Prices, the Cut can still Drastically Increase or Decrease its value.**
# * **With a Higher Cut Quality, the Diamond’s Cost per Carat Increases.**
# * **This is because there is a Higher Wastage of the Rough Stone as more Material needs to be Removed in order to achieve better Proportions and Symmetry.**
# 
# [Click Here to Lean More about How Cut Affects the Price.](https://www.lumeradiamonds.com/diamond-education/diamond-cut)

# ![](https://i.imgur.com/6PannTm.jpg)

# In[ ]:


sns.factorplot(x='cut', data=df , kind='count',aspect=2.5 )


# ## Cut vs Price

# In[ ]:


sns.factorplot(x='cut', y='price', data=df, kind='box' ,aspect=2.5 )


# In[ ]:


# Understanding Box Plot :

# The bottom line indicates the min value of Age.
# The upper line indicates the max value.
# The middle line of the box is the median or the 50% percentile.
# The side lines of the box are the 25 and 75 percentiles respectively.


# ### Premium Cut on Diamonds as we can see are the most Expensive, followed by Excellent / Very Good Cut.

# In[ ]:





# ## 3.3) Color
# * **The Color of a Diamond refers to the Tone and Saturation of Color, or the Depth of Color in a Diamond.**
# * **The Color of a Diamond can Range from Colorless to a Yellow or a Faint Brownish Colored hue.**
# * **Colorless Diamonds are Rarer and more Valuable because they appear Whiter and Brighter.**
# 
# [Click Here to Learn More about How Color Affects the Price](https://enchanteddiamonds.com/education/understanding-diamond-color)

# ![](https://i.imgur.com/Ij090Kn.jpg)

# In[ ]:


sns.factorplot(x='color', data=df , kind='count',aspect=2.5 )


# ### Color vs Price

# In[ ]:


sns.factorplot(x='color', y='price' , data=df , kind='violin', aspect=2.5)


# In[ ]:





# ## 3.4) Clarity
# * **Diamond Clarity refers to the absence of the Inclusions and Blemishes.**
# * **An Inclusion is an Imperfection located within a Diamond. Inclusions can be Cracks or even Small Minerals or Crystals that have formed inside the Diamond.**
# * **Blemishing is a result of utting and polishing process than the environmental conditions in which the diamond was formed. It includes scratches, extra facets etc.**
# 
# [Click Here to Learn More about How Clarity Affects the Price of Diamonds.](https://www.diamondmansion.com/blog/understanding-how-diamond-clarity-affects-value/)

# ![](https://i.imgur.com/fLbAstc.jpg)

# In[ ]:


labels = df.clarity.unique().tolist()
sizes = df.clarity.value_counts().tolist()
colors = ['#006400', '#E40E00', '#A00994', '#613205', '#FFED0D', '#16F5A7','#ff9999','#66b3ff']
explode = (0.1, 0.0, 0.1, 0, 0.1, 0, 0.1,0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=0)
plt.axis('equal')
plt.title("Percentage of Clarity Categories")
plt.plot()
fig=plt.gcf()
fig.set_size_inches(6,6)
print()


# In[ ]:


sns.boxplot(x='clarity', y='price', data=df )


# ### It seems that VS1 and VS2 affect the Diamond's Price equally having quite high Price margin.

# In[ ]:





# ## 3.5) Depth
# * **The Depth of a Diamond is its Height (in millimeters) measured from the Culet to the Table.**
# * **If a Diamond's Depth Percentage is too large or small the Diamond will become Dark in appearance because it will no longer return an Attractive amount of light.**
# 
# [Click Here to Learn More about How Depth Affects the Price of Diamonds.](https://beyond4cs.com/grading/depth-and-table-values/)

# In[ ]:


plt.hist('depth' , data=df , bins=25)


# In[ ]:


sns.jointplot(x='depth', y='price' , data=df , kind='regplot', size=5)


# ### We can Infer from the plot that the Price can vary heavily for the same Depth.
# * **And the Pearson's Correlation shows that there's a slightly inverse relation between the two.**

# In[ ]:





# ## 3.6) Table
# * **Table is the Width of the Diamond's Table expressed as a Percentage of its Average Diameter.**
# * **If the Table (Upper Flat Facet) is too Large then light will not play off of any of the Crown's angles or facets and will not create the Sparkly Rainbow Colors.**
# * **If it is too Small then the light will get Trapped and that Attention grabbing shaft of light will never come out but will “leak” from other places in the Diamond.**
# 
# [Click Here to Learn More about How Table Affects the Price of Diamonds.](https://beyond4cs.com/grading/depth-and-table-values/)

# In[ ]:


sns.kdeplot(df['table'] ,shade=True , color='orange')


# In[ ]:


sns.jointplot(x='table', y='price', data=df , size=5)


# In[ ]:





# ## 3.7) Dimensions

# * **As the Dimensions increases, Obviously the Prices Rises as more and more Natural Resources are Utilised.**

# In[ ]:


sns.kdeplot(df['x'] ,shade=True , color='r' )
sns.kdeplot(df['y'] , shade=True , color='g' )
sns.kdeplot(df['z'] , shade= True , color='b')
plt.xlim(2,10)


# **We'll Create a New Feature based on the Dimensions in the Next Section called 'Volume' and Visualize how it affects the Price.**

# In[ ]:





# <a id="there_you_go_4"></a>
# # 4) Feature Engineering

# ## 4.1) Create New Feature 'Volume'

# In[ ]:


df['volume'] = df['x']*df['y']*df['z']
df.head()


# In[ ]:


plt.figure(figsize=(5,5))
plt.hist( x=df['volume'] , bins=30 ,color='g')
plt.xlabel('Volume in mm^3')
plt.ylabel('Frequency')
plt.title('Distribution of Diamond\'s Volume')
plt.xlim(0,1000)
plt.ylim(0,50000)


# In[ ]:


sns.jointplot(x='volume', y='price' , data=df, size=5)


# ### It seems that there is Linear Relationship between Price and Volume (x \* y \* z).

# ## 4.2) Drop X, Y, Z

# In[ ]:


df.drop(['x','y','z'], axis=1, inplace= True)
#df.head()


# In[ ]:





# <a id="there_you_go_5"></a>
# # 5) Feature Encoding

# * **Label the Categorical Features with digits to Distinguish.**
# * **As we can't feed String data for Modelling.**

# In[ ]:


label_cut = LabelEncoder()
label_color = LabelEncoder()
label_clarity = LabelEncoder()


df['cut'] = label_cut.fit_transform(df['cut'])
df['color'] = label_color.fit_transform(df['color'])
df['clarity'] = label_clarity.fit_transform(df['clarity'])


# In[ ]:


#df.head()


# <a id="there_you_go_6"></a>
# # 6) Feature Scaling

# * **Divide the Dataset into Train and Test, So that we can fit the Train for Modelling Algos and Predict on Test.**
# * **Then Apply Feature Scaling although it's not neccessary in this case. But it surely helps.**

# In[ ]:


# Split the data into train and test.


# In[ ]:


X = df.drop(['price'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=66)


# In[ ]:


# Applying Feature Scaling ( StandardScaler )
# You can also Apply MinMaxScaler.


# In[ ]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:





# <a id="there_you_go_7"></a>
# # 7) Modelling Algos

# In[ ]:


# Collect all R2 Scores.
R2_Scores = []
models = ['Linear Regression' , 'Lasso Regression' , 'AdaBoost Regression' , 'Ridge Regression' , 'GradientBoosting Regression', 'RandomForest Regression' , 'KNeighbours Regression'] 


# ## 7.1) Linear Regression

# In[ ]:


clf_lr = LinearRegression()
clf_lr.fit(X_train , y_train)
accuracies = cross_val_score(estimator = clf_lr, X = X_train, y = y_train, cv = 5,verbose = 1)
y_pred = clf_lr.predict(X_test)
print('')
print('####### Linear Regression #######')
print('Score : %.4f' % clf_lr.score(X_test, y_test))
print(accuracies)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

R2_Scores.append(r2)


# ## 7.2) Lasso Regression

# In[ ]:


clf_la = Lasso(normalize=True)
clf_la.fit(X_train , y_train)
accuracies = cross_val_score(estimator = clf_la, X = X_train, y = y_train, cv = 5,verbose = 1)
y_pred = clf_la.predict(X_test)
print('')
print('###### Lasso Regression ######')
print('Score : %.4f' % clf_la.score(X_test, y_test))
print(accuracies)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

R2_Scores.append(r2)


# ## 7.3) AdaBosst Regression

# In[ ]:


clf_ar = AdaBoostRegressor(n_estimators=1000)
clf_ar.fit(X_train , y_train)
accuracies = cross_val_score(estimator = clf_ar, X = X_train, y = y_train, cv = 5,verbose = 1)
y_pred = clf_ar.predict(X_test)
print('')
print('###### AdaBoost Regression ######')
print('Score : %.4f' % clf_ar.score(X_test, y_test))
print(accuracies)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

R2_Scores.append(r2)


# ## 7.4) Ridge Regression

# In[ ]:


clf_rr = Ridge(normalize=True)
clf_rr.fit(X_train , y_train)
accuracies = cross_val_score(estimator = clf_rr, X = X_train, y = y_train, cv = 5,verbose = 1)
y_pred = clf_rr.predict(X_test)
print('')
print('###### Ridge Regression ######')
print('Score : %.4f' % clf_rr.score(X_test, y_test))
print(accuracies)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

R2_Scores.append(r2)


# ## 7.5) GradientBoosting Regression

# In[ ]:


clf_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls',verbose = 1)
clf_gbr.fit(X_train , y_train)
accuracies = cross_val_score(estimator = clf_gbr, X = X_train, y = y_train, cv = 5,verbose = 1)
y_pred = clf_gbr.predict(X_test)
print('')
print('###### Gradient Boosting Regression #######')
print('Score : %.4f' % clf_gbr.score(X_test, y_test))
print(accuracies)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

R2_Scores.append(r2)


# ## 7.6) RandomForest Regression

# In[ ]:


clf_rf = RandomForestRegressor()
clf_rf.fit(X_train , y_train)
accuracies = cross_val_score(estimator = clf_rf, X = X_train, y = y_train, cv = 5,verbose = 1)
y_pred = clf_rf.predict(X_test)
print('')
print('###### Random Forest ######')
print('Score : %.4f' % clf_rf.score(X_test, y_test))
print(accuracies)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)


# ### Tuning Parameters

# In[ ]:


no_of_test=[100]
params_dict={'n_estimators':no_of_test,'n_jobs':[-1],'max_features':["auto",'sqrt','log2']}
clf_rf=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='r2')
clf_rf.fit(X_train,y_train)
print('Score : %.4f' % clf_rf.score(X_test, y_test))
pred=clf_rf.predict(X_test)
r2 = r2_score(y_test, pred)
print('R2     : %0.2f ' % r2)
R2_Scores.append(r2)


# ## 7.7) KNeighbours Regression

# In[ ]:


clf_knn = KNeighborsRegressor()
clf_knn.fit(X_train , y_train)
accuracies = cross_val_score(estimator = clf_knn, X = X_train, y = y_train, cv = 5,verbose = 1)
y_pred = clf_knn.predict(X_test)
print('')
print('###### KNeighbours Regression ######')
print('Score : %.4f' % clf_knn.score(X_test, y_test))
print(accuracies)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)


# ### Tuning Parameters

# In[ ]:


n_neighbors=[]
for i in range (0,50,5):
    if(i!=0):
        n_neighbors.append(i)
params_dict={'n_neighbors':n_neighbors,'n_jobs':[-1]}
clf_knn=GridSearchCV(estimator=KNeighborsRegressor(),param_grid=params_dict,scoring='r2')
clf_knn.fit(X_train,y_train)
print('Score : %.4f' % clf_knn.score(X_test, y_test))
pred=clf_knn.predict(X_test)
r2 = r2_score(y_test, pred)
print('R2     : %0.2f ' % r2)
R2_Scores.append(r2)


# In[ ]:





# <a id="there_you_go_8"></a>
# # 8) Visualizing R2-Score of Algorithms

# In[ ]:


compare = pd.DataFrame({'Algorithms' : models , 'R2-Scores' : R2_Scores})
compare.sort_values(by='R2-Scores' ,ascending=False)


# In[ ]:


sns.barplot(x='R2-Scores' , y='Algorithms' , data=compare)


# In[ ]:


sns.factorplot(x='Algorithms', y='R2-Scores' , data=compare, size=6 , aspect=4)


# ### Random Forest Regressor gives us the highest R2-Score [ 98% ] .

# # Thank You :)

# In[ ]:




