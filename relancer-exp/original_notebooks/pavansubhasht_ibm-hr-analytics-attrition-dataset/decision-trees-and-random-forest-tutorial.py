#!/usr/bin/env python
# coding: utf-8

# ***
# ### **I hope you find this kernel useful and your <font color="red"><b>UPVOTES</b></font> would be highly appreciated**
# ***
# # Decision Tree & Random Forest
# 
# # 1. Decision Tree
# 
# ![Capture.PNG](attachment:Capture.PNG)
# 
# Decision Trees are an important type of algorithm for predictive modeling machine learning.
# 
# The classical decision tree algorithms have been around for decades and modern variations like random forest are among the most powerful techniques available.
# 
# Classification and Regression Trees or `CART` for short is a term introduced by `Leo Breiman` to refer to Decision Tree algorithms that can be used for classification or regression predictive modeling problems.
# 
# Classically, this algorithm is referred to as “`decision trees`”, but on some platforms like R they are referred to by the more modern term CART.
# 
# The `CART` algorithm provides a foundation for important algorithms like `bagged decision trees`, `random forest` and `boosted decision trees`.
# 
# ### CART Model Representation
# The representation for the CART model is a binary tree.
# 
# This is your binary tree from algorithms and data structures, nothing too fancy. Each root node represents a single input variable (x) and a split point on that variable (assuming the variable is numeric).
# 
# The leaf nodes of the tree contain an output variable (y) which is used to make a prediction.
# 
# Given a new input, the tree is traversed by evaluating the specific input started at the root node of the tree.
# 
# #### Some **advantages** of decision trees are:
# * Simple to understand and to interpret. Trees can be visualised.
# * Requires little data preparation. 
# * Able to handle both numerical and categorical data.
# * Possible to validate a model using statistical tests. 
# * Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.
# 
# #### The **disadvantages** of decision trees include:
# * Overfitting. Mechanisms such as pruning (not currently supported), setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.
# * Decision trees can be unstable. Mitigant: Use decision trees within an ensemble.
# * Cannot guarantee to return the globally optimal decision tree. Mitigant: Training multiple trees in an ensemble learner
# * Decision tree learners create biased trees if some classes dominate. Recommendation: Balance the dataset prior to fitting
# 
# # 2. Random Forest
# Random Forest is one of the most popular and most powerful machine learning algorithms. It is a type of ensemble machine learning algorithm called Bootstrap Aggregation or bagging.
# ![inbox_3363440_e322b7c76f2ca838ba3753e3c76c5efc_inbox_2301650_875af39bcc296f0a783519a400412dee_RF.jpg](attachment:inbox_3363440_e322b7c76f2ca838ba3753e3c76c5efc_inbox_2301650_875af39bcc296f0a783519a400412dee_RF.jpg)
# To improve performance of Decision trees, we can use many trees with a random sample of features chosen as the split.

# # 3. Decision Tree & Random Forest Implementation in python
# 
# We will use Decision Tree & Random Forest in Predicting the attrition of your valuable employees.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


# In[ ]:


df = pd.read_csv("../../../input/pavansubhasht_ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head()


# ## 1. Exploratory Data Analysis

# In[ ]:


df.info()


# In[ ]:


pd.set_option("display.float_format", "{:.2f}".format)
df.describe()


# In[ ]:


df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis="columns", inplace=True)


# In[ ]:


categorical_col = []
for column in df.columns:
    if df[column].dtype == object and len(df[column].unique()) <= 50:
        categorical_col.append(column)
        print(f"{column} : {df[column].unique()}")
        print("====================================")


# In[ ]:


df['Attrition'] = df.Attrition.astype("category").cat.codes


# ## 2. Data Visualisation

# In[ ]:


df.Attrition.value_counts()


# In[ ]:


# Visulazing the distibution of the data for every feature
df.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20))


# In[ ]:


# Plotting how every feature correlate with the "target"
sns.set(font_scale=1.2)
plt.figure(figsize=(30, 30))

for i, column in enumerate(categorical_col, 1):
    plt.subplot(3, 3, i)
    g = sns.barplot(x=f"{column}", y='Attrition', data=df)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    plt.ylabel('Attrition Count')
    plt.xlabel(f'{column}')


# **Conclusions:**
# 
# ***
# - `BusinessTravel` : The workers who travel alot are more likely to quit then other employees.
# 
# - `Department` : The worker in `Research & Development` are more likely to stay then the workers on other departement.
# 
# - `EducationField` : The workers with `Human Resources` and `Technical Degree` are more likely to quit then employees from other fields of educations.
# 
# - `Gender` : The `Male` are more likely to quit.
# 
# - `JobRole` : The workers in `Laboratory Technician`, `Sales Representative`, and `Human Resources` are more likely to quit the workers in other positions.
# 
# - `MaritalStatus` : The workers who have `Single` marital status are more likely to quit the `Married`, and `Divorced`.
# 
# - `OverTime` : The workers who work more hours are likely to quit then others.
# 
# *** 

# ## 3. Correlation Matrix

# In[ ]:


plt.figure(figsize=(30, 30))


# ## 4. Data Processing

# In[ ]:


categorical_col.remove('Attrition')


# In[ ]:


# Transform categorical data into dummies
# categorical_col.remove("Attrition")
# data = pd.get_dummies(df, columns=categorical_col)
# data.info()
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
for column in categorical_col:
    df[column] = label.fit_transform(df[column])


# In[ ]:


X = df.drop('Attrition', axis=1)
y = df.Attrition


# ## 5. Applying machine learning algorithms

# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        print("Train Result:\n===========================================")
        print(f"accuracy score: {accuracy_score(y_train, pred):.4f}\n")
        print(f"Classification Report: \n \tPrecision: {precision_score(y_train, pred)}\n\tRecall Score: {recall_score(y_train, pred)}\n\tF1 score: {f1_score(y_train, pred)}\n")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, clf.predict(X_train))}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        print("Test Result:\n===========================================")        
        print(f"accuracy score: {accuracy_score(y_test, pred)}\n")
        print(f"Classification Report: \n \tPrecision: {precision_score(y_test, pred)}\n\tRecall Score: {recall_score(y_test, pred)}\n\tF1 score: {f1_score(y_test, pred)}\n")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ### 5. 1. Decision Tree Classifier
# 
# **Decision Tree parameters:**
# - `criterion`: The function to measure the quality of a split. Supported criteria are "`gini`" for the Gini impurity and "`entropy`" for the information gain.
# ***
# - `splitter`: The strategy used to choose the split at each node. Supported strategies are "`best`" to choose the best split and "`random`" to choose the best random split.
# ***
# - `max_depth`: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than `min_samples_split` samples.
# ***
# - `min_samples_split`: The minimum number of samples required to split an internal node.
# ***
# - `min_samples_leaf`: The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples in each of the left and right branches.  This may have the effect of smoothing the model, especially in regression.
# ***
# - `min_weight_fraction_leaf`: The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
# ***
# - `max_features`: The number of features to consider when looking for the best split.
# ***
# - `max_leaf_nodes`: Grow a tree with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
# ***
# - `min_impurity_decrease`: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
# ***
# - `min_impurity_split`: Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

print_score(tree, X_train, y_train, X_test, y_test, train=True)
print_score(tree, X_train, y_train, X_test, y_test, train=False)


# ### 5. 2. Decision Tree Classifier Hyperparameter tuning

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

params ={"criterion":("gini", "entropy"),"splitter":("best", "random"),"max_depth":(list(range(1, 20))),"min_samples_split":[2, 3, 4],"min_samples_leaf":list(range(1, 20)),}


model = DecisionTreeClassifier(random_state=42)
grid_search_cv = GridSearchCV(model, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=3)

# grid_search_cv.fit(X_train, y_train)


# In[ ]:


# grid_search_cv.best_estimator_


# In[ ]:


tree = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',max_depth=6, max_features=None, max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=10, min_samples_split=2,min_weight_fraction_leaf=0.0, presort='deprecated',random_state=42, splitter='best')


# In[ ]:


tree.fit(X_train, y_train)


# In[ ]:


print_score(tree, X_train, y_train, X_test, y_test, train=True)
print_score(tree, X_train, y_train, X_test, y_test, train=False)


# ### Visualization of a tree

# In[ ]:


from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydot

features = list(df.columns)
features.remove("Attrition")


# In[ ]:


dot_data = StringIO()
export_graphviz(tree, out_file=dot_data, feature_names=features, filled=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())


# ### 5. 3. Random Forest
# 
# A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
# 
# - **Random forest algorithm parameters:**
# - `n_estimators`: The number of trees in the forest.
# *** 
# - `criterion`: The function to measure the quality of a split. Supported criteria are "`gini`" for the Gini impurity and "`entropy`" for the information gain.
# ***
# - `max_depth`: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than `min_samples_split` samples.
# ***
# - `min_samples_split`: The minimum number of samples required to split an internal node.
# ***
# - `min_samples_leaf`: The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples in each of the left and right branches.  This may have the effect of smoothing the model, especially in regression.
# ***
# - `min_weight_fraction_leaf`: The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
# ***
# - `max_features`: The number of features to consider when looking for the best split.
# ***
# - `max_leaf_nodes`: Grow a tree with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
# ***
# - `min_impurity_decrease`: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
# ***
# - `min_impurity_split`: Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.
# ***
# - `bootstrap`: Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.
# ***
# - `oob_score`: Whether to use out-of-bag samples to estimate the generalization accuracy.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rand_forest = RandomForestClassifier(n_estimators=100)
rand_forest.fit(X_train, y_train)

print_score(rand_forest, X_train, y_train, X_test, y_test, train=True)
print_score(rand_forest, X_train, y_train, X_test, y_test, train=False)


# ### 5. 4. Random Forest hyperparameter tuning

# ### a) Randomized Search Cross Validation

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators, 'max_features': max_features,'max_depth': max_depth, 'min_samples_split': min_samples_split,'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

rand_forest = RandomForestClassifier(random_state=42)

rf_random = RandomizedSearchCV(estimator=rand_forest, param_distributions=random_grid, n_iter=100, cv=3,verbose=2, random_state=42, n_jobs=-1)


# rf_random.fit(X_train, y_train)


# In[ ]:


# rf_random.best_estimator_


# In[ ]:


rand_forest = RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,criterion='gini', max_depth=80, max_features='sqrt',max_leaf_nodes=None, max_samples=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=5,min_weight_fraction_leaf=0.0, n_estimators=1400,n_jobs=None, oob_score=False, random_state=42, verbose=0,warm_start=False)
rand_forest.fit(X_train, y_train)


# In[ ]:


print_score(rand_forest, X_train, y_train, X_test, y_test, train=True)
print_score(rand_forest, X_train, y_train, X_test, y_test, train=False)


# `Random search` allowed us to narrow down the range for each hyperparameter. Now that we know where to concentrate our search, we can explicitly specify every combination of settings to try. We do this with `GridSearchCV`, a method that, instead of sampling randomly from a distribution, evaluates all combinations we define.

# ### b) Grid Search Cross Validation

# In[ ]:


param_grid = {'max_depth':[50, 60, 75],'n_estimators':[1400, 1425, 1450],'max_features':['sqrt'],'min_samples_split':[4, 5, 6],'min_samples_leaf':[1],'bootstrap':[ False],'criterion':["gini"]}

rand_frst_clf = RandomForestClassifier(random_state=42, n_estimators=1000)

grid_rand_forest = GridSearchCV(rand_frst_clf, param_grid, scoring="accuracy",n_jobs=-1, verbose=1, cv=3)


# In[ ]:


# grid_rand_forest.fit(X_train, y_train)


# In[ ]:


# grid_rand_forest.best_estimator_


# In[ ]:


rand_forest = RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,criterion='gini', max_depth=50, max_features='sqrt',max_leaf_nodes=None, max_samples=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=4,min_weight_fraction_leaf=0.0, n_estimators=1400,n_jobs=None, oob_score=False, random_state=42, verbose=0,warm_start=False)
rand_forest.fit(X_train, y_train)


# In[ ]:


print_score(rand_forest, X_train, y_train, X_test, y_test, train=True)
print_score(rand_forest, X_train, y_train, X_test, y_test, train=False)


# # 4. Summary
# In this notebook we learned the following lessons:
# - Decsion tree and random forest algorithms and the parameters of each algorithm.
# - How to tune hyperparameters for both Decision tree and Random Forest.
# - Balance your dataset before training to prevent the tree from being biased toward the classes that are dominant. 
#   - By sampling an equal number of samples from each class  
#   - By normalizing the sum of the sample weights (sample_weight) for each class to the same value. 
# 
#   
# ## References:
# - [Hyperparameter Tuning the Random Forest in Python](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)
# - [Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
# - [Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)
# - [Bagging and Random Forest Ensemble Algorithms for Machine Learning](https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/)

