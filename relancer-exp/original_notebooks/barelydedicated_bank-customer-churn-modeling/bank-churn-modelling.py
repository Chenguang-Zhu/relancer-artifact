#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Modeling
# <img src="https://miro.medium.com/max/1024/1*YRq10sAcj2ScV2TirdSKBg.png" height =500 width=500></imag>
# 
# ##### Can you predict if bank customers will turnover next cycle?
# 
# Churn prevention allows companies to develop loyalty programs and retention campaigns to keep as many customers as possible.

# # Machine Learning Process Step by Step
# <img src="https://miro.medium.com/max/1399/0*C_ibLD-RscbJzjMq.png" height =500 width=500></imag>
# 
# 

# ### Importing Libraries
# - numpy
# - matplotlib
# - seaborn
# - scikit-learn
# - imblearn
# 

# In[ ]:


# Importing Librarys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV # to split the data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score, roc_curve #To evaluate our model
from sklearn.metrics import plot_confusion_matrix
from sklearn.externals import joblib
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

from pylab import rcParams

import warnings
warnings.filterwarnings("ignore")


# ### Loading Dataset
# <a>https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling </a>
# 
# __Given below are columns information__
# 1. RowNumber
# 2. CustomerId
# 3. Surname
# 4. CreditScore
# 5. Geography
# 6. Gender
# 7. Age
# 8. Tenure
# 9. Balance
# 10. NumOfProductsHow many accounts, bank account affiliated products the person has
# 11. HasCrCard
# 12. IsActiveMemberSubjective, but for the concept
# 13. EstimatedSalary
# 14. Exited Did they leave the bank after all?

# In[ ]:


# Using pandas 
# Data downloaded from above link and stored in local folder
dataframe = pd.read_csv("../../../input/barelydedicated_bank-customer-churn-modeling/Churn_Modelling.csv")


# ## Exploratory Data Analysis
# - __Summarizing, Describing and Data Distributions__
# - __Univariate and bivariate Analysis__
# - __Outliers and their influence__
# - __Metadata errors__
# - __Missing Data__
# - __Correlation analysis between variables__
# 

# In[ ]:


dataframe.head(5)


# In[ ]:


# Find out the total number of rows and columns
dataframe.shape


# In[ ]:


#columns and their corresponding data types,along with finding whether they contain null values or not
dataframe.info()


# In[ ]:


# The describe() function in pandas is very handy in getting various summary statistics.
# This function returns the count, mean, standard deviation, minimum and maximum values and the quantiles of the data
dataframe.describe()


# In[ ]:


# Checking data sampling
dataframe.Exited.unique()
dataframe.Exited.value_counts()


# In[ ]:


# Data Visualization
import seaborn as sns
sns.countplot(dataframe['Exited'],label="Count")
print()


# In[ ]:


# Data to plot
male_dataframe = dataframe[dataframe.Gender=="Male"]
sizes = male_dataframe['Exited'].value_counts(sort = True)
colors = ["Green","Red"] 
rcParams['figure.figsize'] = 5,5
# Plot
plt.pie(sizes,  colors=colors, autopct='%1.1f%%', shadow=True, startangle=270,) 
plt.title('Percentage of Churn in Dataset for Male')
print()


# In[ ]:


# Data to plot
female_dataframe = dataframe[dataframe.Gender=="Female"]
sizes = female_dataframe['Exited'].value_counts(sort = True)
colors = ["Green","Red"] 
rcParams['figure.figsize'] = 5,5
# Plot
plt.pie(sizes,  colors=colors, autopct='%1.1f%%', shadow=True, startangle=270,) 
plt.title('Percentage of Churn in Dataset for Female')
print()


# ### Correlation Analysis

# In[ ]:


correlations = dataframe.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
print()
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(dataframe.columns)
ax.set_yticklabels(dataframe.columns)
rcParams['figure.figsize'] = 40,15
print()


# ### Finding Outliers

# In[ ]:


rcParams['figure.figsize'] = 10,10
boxplot = dataframe.boxplot(column=['EstimatedSalary', 'Balance'])


# In[ ]:


rcParams['figure.figsize'] = 10,8
sns.boxplot(x="Geography",y="Exited",data=dataframe,palette='rainbow')


# ## Data Preprocessing
# - __Imputing Missing Data__
# - __Handling Unbalanced Data(Under Sampling and OverSampling)__
# - __Handling Outliers__
# - __Transforming, Encoding, Scaling, and Shuffling__

# ### Imputing Missing Data 

# In[ ]:


# removing null values to avoid errors  
dataframe.dropna(inplace = True)


# ## Feature Engineering
# - __Adding or dropping features__
# - __Combining multiple features into one feature__
# - __Binning__
# - __One Hot Encoding__

# In[ ]:


#remove the fields from the data set that we don't
# want to include in our model
del dataframe['RowNumber']
del dataframe['CustomerId']
del dataframe['Surname']


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


label1 = LabelEncoder()
dataframe['Geography'] = label1.fit_transform(dataframe['Geography'])


# In[ ]:


label2 = LabelEncoder()
dataframe['Gender'] = label2.fit_transform(dataframe['Gender'])


# In[ ]:


features_dataframe = pd.get_dummies(dataframe, columns=['Geography'])


# In[ ]:


#Remove the Exited from the feature data
del features_dataframe['Exited']


# In[ ]:


X = features_dataframe.values
y = dataframe['Exited']
features_dataframe.columns


# ### Handling Imbalanced Data

# In[ ]:


# apply near miss
import imblearn
from imblearn.under_sampling import NearMiss 
nr = NearMiss() 
X, y = nr.fit_sample(X, y)
X[100]


# In[ ]:


y[100]


# In[ ]:


#Split the data set in a traning set (80%) and a test set(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)


# ## Modelling

# ### Training and Model Selection

# In[ ]:


# Algorithmns models to be compared
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[ ]:




# ### Choosed GradientBoostingClassifier Algorithm and List Metrics 

# In[ ]:


# From above models we can see, 
# We are getting highest accuracy and better values of precision and recall for XGBClassifier
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
print('Accuracy = ',accuracy_score(y_test, model.predict(X_test)))
print('classification_report = ',classification_report(y_test, model.predict(X_test)))


# ### Hyper Parameter Tuning

# In[ ]:




# ### Train model based on best hyper parameter

# In[ ]:




# ## Model Evaluation

# #### Accuracy and Classification_report

# In[ ]:


print('Accuracy = ',accuracy_score(y_test, model.predict(X_test)))
print('classification_report = ',classification_report(y_test, model.predict(X_test)))


# ### Plot Confusion Matrix

# In[ ]:


# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None), ("Normalized confusion matrix", 'true')] 
for title, normalize in titles_options:
    disp = plot_confusion_matrix(model, X_test, y_test, display_labels=['Exited','Not Exited'], cmap=plt.cm.Blues, normalize=normalize) 
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

print()


# ### ROC Curve

# In[ ]:


y_pred_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
print()


# In[ ]:





# ### Save Model for Production Deployment

# In[ ]:


# Save the trained model to a file so we can use it in other programs
joblib.dump(model,"customer_churn_mlmodel.pkl")


# ### List Important Features

# In[ ]:


# These are the features labels from out data set
feature_labels = np.array(['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_0','Geography_1', 'Geography_2']) 


# In[ ]:


# Create a numpy array based on the model's feature importances
importance = model.feature_importances_


# In[ ]:


# Sort the feature labels based on the feature importance rankings from the model
feature_indexes_by_importance = importance.argsort()


# In[ ]:


# Print each feature label, from most important to least important (reverse order)
for index in feature_indexes_by_importance:
    print("{} - {:.2f}%".format(feature_labels[index], (importance[index] * 100.0)))


# In[ ]:




