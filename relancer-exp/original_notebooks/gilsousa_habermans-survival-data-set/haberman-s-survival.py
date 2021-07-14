#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning and Predictions Survival's

# In[ ]:




import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from matplotlib.ticker import FormatStrFormatter


# **2 Loading the data**
# 
# Relevant Information: The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# 
# Attribute Information:
#    
#    1. Age of patient at time of operation (numerical)
#    
#    2. Patient's year of operation (year - 1900, numerical)
#    
#    3. Number of positive axillary nodes detected (numerical)
#    
#    4. Survival status (class attribute)
#         
#         1 = the patient survived 5 years or longer
#         
#         2 = the patient died within 5 year
#         
# **Data may be found in** http://mlr.cs.umass.edu/ml/machine-learning-databases/haberman/haberman.data

# In[ ]:


#1. Age of patient at time of operation (numerical)
#2. Patient's year of operation (year - 1900, numerical)
#3. Number of positive axillary nodes detected (numerical)
#4. Survival status (class attribute)
         #1 = the patient survived 5 years or longer
         #2 = the patient died within 5 year
        
# Load dataset
url = "../../../input/gilsousa_habermans-survival-data-set/haberman.csv"
names = ['Age', 'Year operation', 'Axillary nodes detected', 'Survival status']
dataset = pandas.read_csv(url, names=names)


# **3 Data preparation**
# 
# In this phase you enhance the quality of the
# data and prepare it for use in subsequent steps. Data transformation ensures that the data is in a suitable
# format for use in your models.

# In[ ]:


dataset.head(5)


# In[ ]:


dataset.describe()


# **4 Data exploration**
# 
# Data exploration is concerned with building a deeper understanding of your data.
# You try to understand how variables interact with each other, the distribution of the
# data, and whether there are outliers. To achieve this you mainly use descriptive statistics,
# visual techniques, plots and simple modeling. 

# In[ ]:


dataset.plot()
print()


# In[ ]:


# histograms
dataset.hist()
print()


# ** 5 Data modeling or model building**
# 
# In this phase you use models, domain knowledge, and insights about the data you
# found in the previous steps to answer the research question. You select a technique
# from the fields of statistics, machine learning, operations research, and so on. Building
# a model is an iterative process that involves selecting the variables for the model,
# executing the model, and model diagnostics. 
# 
# **The modeling phase consists of four steps:**
# 
# A model consists of constructs of information called features or predictors and a target
# or response variable. Your model’s goal is to predict the target variable, for example,
# tomorrow’s high temperature. The variables that help you do this and are (usually)
# known to you are the features or predictor variables such as today’s temperature, cloud
# movements, current wind speed, and so on. The best models are those that accurately
# represent reality, preferably while staying concise and interpretable.
# 
# **5.1** Training the model
# 
# **5.2** Feature engineering and model selection
# 
# **5.3** Model validation and selection
# 
# **5.4** Applying the trained model to unseen data

# **5.1 Training the model**
# 
# With the right predictors in place and a modeling technique in mind, you can progress
# to model training. In this phase you present to your model data from which it
# can learn.

# In[ ]:


#I made an adaptation of this reference online 
#----> http://machinelearningmastery.com/machine-learning-in-python-step-by-step/

array = dataset.values
X = array[:,:3]
Y = array[:,3]
validation_size = 0.30
seed = 10
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed) 


# In[ ]:


#I made an adaptation of this reference online 
#----> http://machinelearningmastery.com/machine-learning-in-python-step-by-step/


# Test options and evaluation metric
num_folds = 20
num_instances = len(X_train)
seed = 10
scoring = 'accuracy'


# **5.2 Model selection**

# In[ ]:


#I made an adaptation of this reference online 
#----> http://machinelearningmastery.com/machine-learning-in-python-step-by-step/



# Spot Check Algorithms
algorithms = []
algorithms.append(('LR', LogisticRegression()))
algorithms.append(('LDA', LinearDiscriminantAnalysis()))
algorithms.append(('KNN', KNeighborsClassifier()))
algorithms.append(('CART', DecisionTreeClassifier()))
algorithms.append(('NB', GaussianNB()))
algorithms.append(('SVM', SVC()))
algorithms.append(('NN', MLPClassifier()))
algorithms.append(('RFC', RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []
for name, algorithm in algorithms:
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    cv_results = cross_validation.cross_val_score(algorithm, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# **5.3 Model validation**
# 
# Data science has many modeling techniques, and the question is which one is the
# right one to use. A good model has two properties: it has good predictive power and it
# generalizes well to data it hasn’t seen. 

# In[ ]:


# Make predictions on validation dataset
knn =  GaussianNB()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# **5.4 Applying the trained model to unseen data**
# 
# First, you prepare a data set that has features
# exactly as defined by your model. This boils down to repeating the data preparation
# you did in step one of the modeling process but for a new data set. Then you apply the
# model on this new data set, and this results in a prediction. 
# 
# 
# **Here I've created a dataframe with random attribute values to use as input to the model.**

# In[ ]:


df_data = {'Age': [30,34, 35,38,40,50,43,45,34,34,46,50,45,38,42], 'Year os operations': [65,64,63,64,66,64,64,64,63,63,64,67,64,65,67], 'axillary nodes detected': [4,10,15,8,40,25,23,40,3,40,3,1,4,2,4]} 
df = pandas.DataFrame(df_data)
print(df)


# In[ ]:


df.plot()
print()


# In[ ]:


prediction = knn.predict(df)


# In[ ]:


print("Prediction of data survival status: {}".format(prediction))


# ** Variable Output (Status survival)**

# In[ ]:


plt.plot(prediction)
plt.ylabel('Status survival')
plt.xlabel('index = number of Occurrences')
print()



# **Conclusion**
# 
# The machine learning algorithm predicts the status of survival with an index of 75% with input attributes: age, year of operation and number of axillary nodes. Remembering that 1 = the patient survived 5 years or longer and
# 2 = the patient died within 5 year.

# **Thanks!**
