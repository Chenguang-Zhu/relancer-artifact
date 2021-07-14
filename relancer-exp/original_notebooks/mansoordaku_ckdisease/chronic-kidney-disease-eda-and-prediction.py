#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# Chronic Kidney disease is the gradual loss of function of the kidney with no symptoms being manifested. <sup>[1](https://en.wikipedia.org/wiki/Chronic_kidney_disease)</sup>  It's difficult to know the burden of the disease since they are no accurate diagnostic tests according to research done [here](https://www.sciencedirect.com/science/article/pii/S2214109X14700026). It could be characterized by [uremic frost](https://en.wikipedia.org/wiki/Uremic_frost); however,  careful diagnosis of the condition should be followed such as testing kidney function URI scan dripstick test for example the specific gravity -- low values(1.01 - 1.010) could mean that the patient has kidney damage, observation of the urine using microscopy and identification of [casts](https://slideplayer.com/slide/4381644/14/images/36/URINE+ANALYSIS+Microscopic+Examination+(Casts)) and other tests can help make a proper diagnosis.  
# 
# In this notebook, we'll use data with 25 features that could be indicative of chronic kidney disease to see if predictive modelling could help us figure out which patients have chronic kidney disease. You can read more about the dataset using this [link](https://www.kaggle.com/mansoordaku/ckdisease). Let's proceed to exploratory data analysis.
# 

# I first import all the packages that could be useful in wrangling, visualization and statistical modelling. I apologise if there's a package here that I have imported but I haven't used it. It may have slipped my mind for some reason.

# In[ ]:


import numpy as np # numeric processing
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split  
from IPython.display import HTML
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from functools import *
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


# Turicreate is a machine learning library by apple. There's some functionality that I was interested in that I wanted to try. In future, I may add it to the notebook. If you want more information about the library.  Find for information [here.](https://github.com/apple/turicreate)

# In[ ]:


print()


# ## Loading data and Exploratory data analysis
# In this analysis, we'll do predictive modelling in hopes of finding a model which will be able to classify the patients appropriately.

# In[ ]:


# load the dataset with pandas read_csv function
df = pd.read_csv("../../../input/mansoordaku_ckdisease/kidney_disease.csv", index_col="id")

# give the dtypes of the columns if the data was squeeky clean
dtypes = { 'id' : np.int32, 'age' : np.int32, 'bp' : np.float32, 'sg' : np.object,  'al' : np.object,  'su' : np.object,  'rbc' : np.object,  'pc' : np.object,  'pcc' : np.object,  'ba' : np.object,  'bgr' : np.float32, 'bu' : np.int32, 'sc' : np.float32, 'sod': np.int32, 'pot' : np.float32, 'hemo' : np.float32, 'pcv' : np.int32, 'wc' : np.int32, 'rc' : np.int32, 'htn' : np.object, 'dm' : np.object, 'cad' : np.object, 'appet': np.object, 'pe' : np.object, 'ane' : np.object, 'class': np.object} 

# another way of reading in the datasets especially very big files like 1GB big
# df2 = dd.read_csv("../../../input/mansoordaku_ckdisease/kidney_disease.csv", dtype=dtypes)
# id                400 non-null int64
# age               391 non-null float64
# bp                388 non-null float64
# sg                353 non-null float64
# al                354 non-null float64
# su                351 non-null float64
# rbc               248 non-null object
# pc                335 non-null object
# pcc               396 non-null object
# ba                396 non-null object
# bgr               356 non-null float64
# bu                381 non-null float64
# sc                383 non-null float64
# sod               313 non-null float64
# pot               312 non-null float64
# hemo              348 non-null float64
# pcv               330 non-null object
# wc                295 non-null object
# rc                270 non-null object
# htn               398 non-null object
# dm                398 non-null object
# cad               398 non-null object
# appet             399 non-null object
# pe                399 non-null object
# ane               399 non-null object
# classification    400 non-null object


# In[ ]:


# see the first couple of observations and transpose 10 observations
# think of it as rolling over your dataset
df.head(10).transpose()


# In[ ]:


# see the column names  
df.columns


# In[ ]:


# see a concise summary of the dataset
df.info()


# * 26 columns and a variable number of observations per feature/variable
# 
# * 400 rows for each id - there could be missing data among the rows of the variable

# In[ ]:


# display summary statistics of each column
# this helps me confirm my assertion on missing data
df.describe(include="all").transpose()


# In[ ]:


# Looking at variables interractively 
profile = ProfileReport(df)

profile


# The good news is that we can work with the current state of the columns since they have been labelled consistently. Bad news is that we have a lot of missing data in this dataset. Let's proceed and find out the number of missing values per column and if the classes are balanced or unbalanced. The profiler did the work already but sometimes it is good to confirm it your own way.

# In[ ]:


# looking for the number of missing observations 
# In the code below a boolean is being tried on each observation asking if the observation is missing or not
# then add all instances of NaN(Not a number) 
missing_values = df.isnull().sum()

# calculating the percentage of missing values in the dataframe
# simply taking the sum of the values we got above dividing by the no of observations in the df
# you could use len(df) instead df.index.size
missing_count_pct = ((missing_values / df.index.size) * 100)

# see how many observations are missing
print(missing_count_pct)


# In[ ]:


# take the missing count percentage and use boolean mask to filter out columns 
# whose observation threshold is greater than 25 percent 
columns_to_drop = missing_count_pct[missing_count_pct > 25].index

# remove columns that meet that threshold and save result in column df_dropped
df_dropped = df.drop(columns_to_drop, axis=1)


# In[ ]:


# number of columns remaining after filtering
df.columns.size - df_dropped.columns.size

# only three columns are lost


# I really hate losing a few columns. I won't throw everything away. But, I will keep these columns while we are doing predictive modelling use the different variants of the datasets and see if there will be any boost in results. In the meantime, let's look at the code book to come up with a hypothesis to find out  which columns are the most important and converting the types of each column to another format that will speed up computation during training.

# In[ ]:


# look at the code book on kaggle and write which columns could be useful here


# According to the original site where we found data [here.](https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease). I found the identity of the columns rather what the columns mean. I'll put a star on the columns i think are important from my background in medical laboratory science. Then the second run through this notebook we could explore only the columns i think are important and lastly use a technique called singular value decomposition to figure out which ones are the most important.
# 
# age - age
# 
# bp - blood pressure *
# 
# sg - specific gravity *
# 
# al - albumin *
# 
# su - sugar *
# 
# rbc - red blood cells *
# 
# pc - pus cell*
# 
# pcc - pus cell clumps *
# 
# ba - bacteria*
# 
# bgr - blood glucose random
# 
# bu - blood urea*
# 
# sc - serum creatinine
# 
# sod - sodium
# 
# pot - potassium
# 
# hemo - hemoglobin*
# 
# pcv - packed cell volume
# 
# wc - white blood cell count*
# 
# rc - red blood cell count*
# 
# htn - hypertension*
# 
# dm - diabetes mellitus*
# 
# cad - coronary artery disease*
# 
# appet - appetite*
# 
# pe - pedal edema*
# 
# ane - anemia*
# 
# class - class* 

# In[ ]:


# checking the types of the column to figure out the best next steps of conversion of data types
df.dtypes


# Review the columns from the original codebook to determine the datatypes then make a schema which we can follow as i import the dataset

# In[ ]:


# fix the columns to be of the categorical type
# if the value is missing replace the NA with the word miss
constant_imputer = SimpleImputer(strategy="constant", fill_value = "miss")

# apply it to categorical columns
df[["rbc"]] = constant_imputer.fit_transform(df[["rbc"]])
df[["pcc"]] = constant_imputer.fit_transform(df[["pcc"]])

# converting the types to be categorical
# Go ahead and use a function here
df['rbc'] = df['rbc'].astype("category")
df['pc'] = df['pc'].astype("category")
df["pcc"] = df['pcc'].astype("category")
df['ba'] = df['ba'].astype("category")
df['appet'] = df['appet'].astype("category")
df['pe'] = df['pe'].astype("category")
df['ane'] = df['ane'].astype("category")
df['classification'] = df['classification'].astype("category")
df['htn'] = df['htn'].astype("category")
df['dm'] = df['dm'].astype("category")
df['cad'] = df['cad'].astype("category")


# confirm the dtypes now
df.dtypes


# In[ ]:


# seeing the columns in list form thinking mode
df.columns


# In[ ]:


# make a copy of the whole dataset
df_copy = df.copy()

# remove the target column for the other uses in the next steps
#df = df.drop("classification", axis = 1)


# In[ ]:


# using a boolean to figure out which columns are of type object and numeric to do other preprocessing 
# in the workflow
object_columns = df.dtypes == "object"
numeric_columns = df.dtypes == "float64"
category_columns = df.dtypes == "category"


# In[ ]:


# use regular expressions to fix it and this is supposed to be one of the first steps after df.dtypes command. If it's categorical 
# you can replace it with anything you want here I use -999 to replace the data entries the tab character to flag them as outliers.
# I change the dtypes so to 32 bit to save memory

df['pcv'] = df['pcv'].replace("\t?",-999).fillna(0).astype("int32") # use  str.replace on column to something meaningful
df['wc'] = df['wc'].replace("\t?", -999).fillna(0).astype("int32") # use  str.replace on column to something meaningful
df['rc'] = df['rc'].replace("\t?", -999).fillna(0).astype("float32") # use  str.replace on column to something meaningful

# exploring another imputation strategy that uses the median 
# mean_imputer = SimpleImputer(strategy="median")
# df["pcv"] = mean_imputer.fit_transform(df["pcv"])
# df["wc"] = mean_imputer.fit_transform(df["wc"])
# df["rc"] = mean_imputer.fit_transform(df["rc"])


# In[ ]:


# write code to extract columns of the type object and numeric
# Make a boolean mask for categorical columns
cat_mask_obj = (df.dtypes == "object") | (df.dtypes == "category")

# Get list of categorical column names
cat_mask_object = df.columns[cat_mask_obj].tolist()

# now for numerical columns
# anything that was parsed as float64 is numeric: make a boolean mask for that
cat_mask_numeric = (df.dtypes == "float64")
cat_mask_numeric = df.columns[cat_mask_numeric].tolist()

# see the result in a combined list: to the left categorical and the right we have numeric columns
print(cat_mask_object, "\n", cat_mask_numeric)


# In[ ]:


# convert all instances of the float 64 to float 32 to speed up computation in the subsequent steps
# remove all the missing values and make sure that they are all numeric
numeric_columns_float32 = df[cat_mask_numeric].astype("float32").fillna(0)


# In[ ]:


#it's worked
numeric_columns_float32.dtypes


# In[ ]:


# Task: split the category columns and object columns to the right type
# you can import the dataset to have the right types too upon import
# you can do this the next time you have time to continue working on this add it as comment though


# They are some columns that are wrongly parsed due to the NAs. They include: pcv(numerical int32), rc(numerical int32). I can either interpolate the missing columns depending how they'll look like in a plot or use mean/median to find the value.

# In[ ]:


# it makes sense that they are some individuals who were not sampled therefore filling the whole dataset with NAs makes sense
# these two columns have data entry problems
# use regular expressions to fix it and this is supposed to be one of the first steps after df.dtypes command. If it's categorical 
# you can replace it with anything you want
#df['pcv'] = df['pcv'].fillna(0, inplace = True)
#df['rc'] = df['rc'].fillna(0, inplace = True)
#df['wc'] = df['wc'].fillna(0, inplace = True)


# In[ ]:


# finding the number of null or NA values in the columns
pd.isnull(df).sum()


# In[ ]:


# checking the dtypes once more 
df.dtypes


# In[ ]:


# concatentate the numeric columns with the category columns to build the full dataset and then X and Y
# remove
df[cat_mask_object] = constant_imputer.fit_transform(df[cat_mask_object])


# In[ ]:


# check for missing values
print(df[cat_mask_object].isnull().sum())
print("*" * 100)
print(numeric_columns_float32.isnull().sum())


# In[ ]:


# bring the columns together with pd.concat
df_clean = pd.concat([numeric_columns_float32, df[cat_mask_object]], axis = 1)

# check the shape of the columns
df_clean.shape


# In[ ]:


# just see the first 10 observations
df_clean.head(10)
# HTML(df_clean.to_html()) see the whole dataframe in HTML format


# In[ ]:


# now see the bottom 10
df_clean.tail(10)


# In[ ]:




# In[ ]:


# some further cleaning is required to remove the \t characters is a couple of columns replacing the instances with the standard formating 
# classification, cad,  dm
df_clean['classification'] = df_clean['classification'].replace("ckd\t","ckd")
df_clean['cad'] = df_clean['cad'].replace("\tno","no")
df_clean['dm'] = df_clean['dm'].replace("\tno","no")
df_clean['dm'] = df_clean['dm'].replace("\tyes", "yes")
df_clean['dm'] = df_clean['dm'].replace(" yes", "yes")


# In[ ]:



# subsetting columns with another boolean mask for categorical columns and object columns
cat_mask_obj2 = (df_clean.dtypes == "object") | (df_clean.dtypes == "category")

# Get list of categorical column names
cat_mask_object2 = df_clean.columns[cat_mask_obj2].tolist()

# remove the column classification 
cat_mask_object2.remove("classification")

# see what columns are left
print(cat_mask_object2)


# In[ ]:


# look into the XGBoost course to figure out how the categorical imputer works
# combine everything and use DictVectorizer for one hot encoding and label encoding

# conversion of our dataframe into a dictionary so as to use DictVectorizer
# this function is mostly used in text processing
df_dict = df_clean[cat_mask_object2].to_dict("records")

# Make a DictVectorizer: use documentation to learn how it works
# In short, it speeds up one hot encoding with meaningful columns created
# we don't want a sparse matrix right?
dv = DictVectorizer(sparse = False)

# Apply fit_transform to our dataset
df_encoded = dv.fit_transform(df_dict)

# see 10 rows
print (df_encoded[:10,:])
print ("=" * 100) # just formatting to distinguish outputs

# print the vocabulary that is, the columns of the dataset, note that order changes
# upon transformation
print(dv.vocabulary_)
print ("=" * 100) # more formatting

print(df_encoded.shape) # number of rows and columns for the encoded dataset
print(df_clean[cat_mask_object2].shape) # number of rows and columns for the original dataset
print("After doing the transformation the columns increase to 21.")


# In[ ]:


# You can try
# make a pipeline to merge the encoding as well as the visualization
# Use t-SNE and or PCA to see the differences between groups this will be the EDA step 
# make a train and test split go through the slides for how to win kaggle competitions and test the ideas
# make the next step a pipeline object like in the xgboost course and try random forest, xgboost and decision tree classifier
# later use the ensembling techniques: Try all the ensembling techniques you know.


# In[ ]:


# see the transformed dataframe with all the missing values imputed
df_clean[cat_mask_numeric]


# In[ ]:


# simply taking the vectorized columns and the numeric columns and bringing them together
# to make an array for a classifier
concat_cols = np.hstack((df_encoded, df_clean[cat_mask_numeric].values))

# another version that is in dataframe format
# make a dataframe with the encoded features and give the columns names from the dictVectorizer
df_cat_var = pd.DataFrame(df_encoded, columns=dv.get_feature_names())

# combine the columns together with the categorical features i.e add columns to the numerical dataframe with other dataframe with the categorical and object data types
concat_cols_df = pd.concat([df_clean[cat_mask_numeric], df_cat_var], axis=1)
concat_cols.shape


# In[ ]:


# the final dataframe we'll use for classification
concat_cols_df


# In[ ]:


# now get the target variable into a numeric form
# there's a simpler step where you can use map instead  y = df_clean["classification"].map(lambda val1: 1 if val1 == "ckd" else 0)
# y = y.values
col_preprocess = df_clean["classification"].replace("ckd", 1) 
final_col_preprocess = col_preprocess.replace("notckd", 0)
y = final_col_preprocess.values
print(y)


# In[ ]:


# confirm if the shape of the vector and matrix
print(concat_cols.shape)
print(y.shape)


# In[ ]:


# now that we have both matrices we can see the distribution of the target variable to know what to do next
# when it comes to preprocessing 
final_col_preprocess.reset_index()["classification"].value_counts(normalize=True)


# In[ ]:


# The patients with chronic kidney disease are more than those who don't have .63 ckd and .38 for nonckd
# The dataset is imbalanced, we can't use the regular accuracy as an evaluation metric instead confusion matrix and F1 score
# we don't want more of the train set in either so I guess stratify is good option
# I changed the 0.5:0.5 to 0.75:0.25
x_train, x_test, y_train, y_test = train_test_split(concat_cols_df, y, test_size = 0.25, stratify = y, random_state=1243)


# In[ ]:


# Check if the dimensionality is the same for the feature and target set (train)
print("Is the number of rows the same between the features and the target?") 
assert x_train.shape[0] == y_train.shape[0]
print (True)


# In[ ]:


# Check if the dimensionality is the same for the feature and target set (test)
print("Is the number of rows the same between the features and the target?") 
assert x_test.shape[0] == y_test.shape[0]
print (True)


# In[ ]:


# Now checking if the target variable is balanced in the train set
pd.Series(y_train).value_counts(normalize=True)


# In[ ]:


# They are still unbalanced now. Therefore, will have to use the f1 score and change the class weight of the algorithms used like logistic regression


# In[ ]:


# look at the instances of the labels 0 and 1 
pd.Series(y_test).value_counts(normalize=True)


# In[ ]:


# normal scikit learn paradigm of specify, fit and predict for logistic regression or continuos perceptron
clf_lr1 = LogisticRegression(class_weight="balanced", random_state=1243)

clf_lr1.fit(x_train,y_train)

preds1 = clf_lr1.predict(x_test)

# using f1 score instead of other metrics
score_vote1 = f1_score(preds1, y_test)
print('F1-Score: {:.3f}'.format(score_vote1))

# Calculate the classification report
report1 = classification_report(y_test, preds1,target_names=["notckd", "ckd"])
print(report1)


# In[ ]:


# Specify the Decision tree classifier: asks a bunch of if-else statements to come up with a decision 
clf_dt2 = DecisionTreeClassifier(class_weight = "balanced",random_state=1243)

clf_dt2.fit(x_train,y_train)

preds2 = clf_dt2.predict(x_test)

score_vote2 = f1_score(preds2, y_test)
print('F1-Score: {:.3f}'.format(score_vote2))

# Calculate the classification report
report2 = classification_report(y_test, preds2, target_names=["notckd", "ckd"])
print(report2)


# In[ ]:


# normalize data functions
# log, z scores(standardized_data), dimensionality reduction with PCA(dim_reduction) and making a function that combines another function(compose2)
def skew (data):
    skewed_data = np.log(data)
    return skewed_data


def standardized_data(data):
    scaler = StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data

def dim_reduction(data):
    pca = PCA(n_components=2)
    return pca.fit_transform(data)


def compose2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))

def compose(*fs):
    return reduce(compose2, fs)


# returns an error due to the points being too small
normalize_data = compose2(skew, standardized_data)

# mean of 0 and a std 1 for all the columns
scaled_x_train = standardized_data(x_train)
scaled_x_test = standardized_data(x_test)

# reduce dimensionality to 2
transform_data = compose2(standardized_data, dim_reduction)
dim_red_x_train = transform_data(x_train)
dim_red_x_test = transform_data(x_test)


# In[ ]:


# Logistic regression just a modified perceptron algorithm that uses the sigmoid function therefore an continous perceptron
clf_lr3 = LogisticRegression(class_weight="balanced",random_state=1243)

clf_lr3.fit(scaled_x_train,y_train)

preds3 = clf_lr3.predict(scaled_x_test)

score_vote3 = f1_score(preds3, y_test)
print('F1-Score: {:.3f}'.format(score_vote3))

# Calculate the classification report
report3= classification_report(y_test, preds3,target_names=["notckd", "ckd"])
print(report3)


# In[ ]:


# Logistic regression but the features have been compressed
clf_lr5 = LogisticRegression(class_weight="balanced",random_state=1243)

clf_lr5.fit(dim_red_x_train,y_train)

preds5 = clf_lr5.predict(dim_red_x_test)

score_vote5 = f1_score(preds5, y_test)
print('F1-Score: {:.3f}'.format(score_vote5))

# Make a classification report
report5 = classification_report(y_test, preds5,target_names=["notckd", "ckd"])
print(report5)


# In[ ]:


# Decision tree classifier but with scaled features
clf_dt4 = DecisionTreeClassifier(class_weight="balanced", random_state=1243)

clf_dt4.fit(scaled_x_train,y_train)

preds4 = clf_dt4.predict(scaled_x_test)

score_vote4 = f1_score(preds4, y_test)
print('F1-Score: {:.3f}'.format(score_vote4))

# Make a classification report
report4 = classification_report(y_test, preds4, target_names=["notckd", "ckd"],)
print(report4)


# In[ ]:


# take the coefficient and see the dimension
clf_lr1.coef_.shape


# In[ ]:


# helps with visualizing the decision function for the classifier
def plot_points(features, labels):
    '''  ''' 
    X = np.array(features) # convert data into an numpy array: features
    y = np.array(labels) # convert data into an numpy array: labels
    ckd = X[np.argwhere(y==1)] # get all instances where the features are for individuals with ckd
    notckd = X[np.argwhere(y==0)] # get all instances where the features are for individuals without ckd
    plt.scatter([s[0][0] for s in ckd], [s[0][1] for s in ckd], s = 30, color = 'cyan', edgecolor = 'k', marker = '^') 
    plt.scatter([s[0][0] for s in notckd], [s[0][1] for s in notckd], s = 30, color = 'red', edgecolor = 'k', marker = 's') 
    plt.xlabel('aack')
    plt.ylabel('beep')
    plt.legend(['ckd','notckd'])
def draw_line(a,b,c, color='black', linewidth=2.0, linestyle='solid', starting=0, ending=3):
    # Plotting the line ax + by + c = 0
    x = np.linspace(starting, ending, 1000)
    plt.plot(x, -c/b - a*x/b, linestyle=linestyle, color=color, linewidth=linewidth)


# In[ ]:


# Trying to visualize the function but this didn't work so well
X = np.array(concat_cols)
y = np.array(y)
ckd = X[np.argwhere(y==0)]
notckd = X[np.argwhere(y==1)]

plt.scatter([s[0][0] for s in ckd], [s[0][1] for s in ckd], s = 25, color = 'cyan', edgecolor = 'k', marker = '^') 
plt.scatter([s[0][0] for s in notckd], [s[0][1] for s in notckd], s = 25, color = 'red', edgecolor = 'k', marker = 's') 
plt.xlabel('ckd')
plt.ylabel('notckd')
plt.legend(['ckd','notckd'])


# In[ ]:


# This needs some fixing: Please ignore this for now.
plot_points(scaled_x_train, y_train)
draw_line(1,1, clf_lr1.fit_intercept)


# In[ ]:


# Check this out https://github.com/luisguiserrano/manning/blob/master/Chapter%205%20-%20Logistic%20Regression/Coding%20the%20Logistic%20Regression%20Algorithm.ipynb


# In[ ]:


# plotting feature importance for the Decision tree
# grab the column names as a list
features = concat_cols_df.columns

# get the feature importances
important_features = clf_dt2.feature_importances_

# find the indices of a sorted array
feature_indices = np.argsort(important_features)

# make a plot 
plt.title('Feature Importances Decision Tree')
plt.xticks(fontsize=6, rotation = 45)
plt.barh(range(len(feature_indices)), important_features[feature_indices], color='g', align='center')
plt.yticks(range(len(feature_indices)), [features[i] for i in feature_indices], fontsize = 6)
plt.xlabel('Relative Importance')
print()


# In[ ]:


# Reviewing feature importance using the logistic regression and the C parameter
# grab the coefficients and transpose the array
# label the C parameter
plt.plot(np.sort(clf_lr1.coef_.T), 'o', label="C=1",color = "g") 
plt.xticks(range(concat_cols_df.shape[1]), concat_cols_df.columns, rotation=90)
plt.hlines(0, 0, concat_cols_df.shape[1])
plt.title("Examination of feature importance")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend()


# Common features of importance between the two models: sg(specific gravity), dm=yes(diabetes mellitus), age

# In[ ]:


print()


# In[ ]:


# draw the decision tree 
# add more comments for this
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf_dt2, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names = concat_cols_df.columns, class_names =["notckd", "ckd"]) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

# look at hemo(hemoglobin), sg(specific gravity), al(albumin), sod(sodium), rbc=normal(red blood cells), htn=yes(hypertension), bu(blood urea)
# dm (diabetes mellitus)


# In[ ]:


# interpreting the logistic regression model
clf_lr1.predict(x_test[:1])


# In[ ]:


# checking out the intercept this means that if every feature is 0 t
clf_lr1.intercept_


# In[ ]:


# checking out the coefficients we'll multiply this by the every value that corresponds to that feature e.g clf_lr1.coef[0] * age[0]
clf_lr1.coef_


# In[ ]:


# checking out the available columns
concat_cols_df.columns


# In[ ]:


# make a dataframe to help easily grab the coefficients for writing the formula and visualizing the data
# transpose to see it clearly 
important_features2 = clf_lr1.coef_
column_coef = pd.DataFrame(list(zip(important_features2.T.ravel("C").tolist(), features)),columns = ["coefficient", "feature"])
column_coef["coefficient"] = column_coef["coefficient"].astype("float32")
column_coef.T


# In[ ]:


x_test[:1] # see all the features in the first column 


# In[ ]:


x_test.shape # see the number of rows and columns


# In[ ]:


# writing the logistic regression formula
# Bo intercept, B1 x1n
# writing the denominator based on the wikipedia entry https://en.wikipedia.org/wiki/Logistic_regression
# 1/ np.exp(-(weight * coefn + clf_lr1.intercept_))
weights_int_bias = clf_lr1.intercept_ + (column_coef.coefficient[0] * 60.0) + (column_coef.coefficient[1] * 60.0) + (column_coef.coefficient[2] * 1.01) + (column_coef.coefficient[3] + 3.0) + (column_coef.coefficient[4] + 1.0) + (column_coef.coefficient[5] * 288) + (column_coef.coefficient[6] * 36.0) + (column_coef.coefficient[7] * 1.7) + (column_coef.coefficient[8] * 130) + (column_coef.coefficient[9] * 3.0) + (column_coef.coefficient[10] * 7.9) + (column_coef.coefficient[11] * 0.0) + (column_coef.coefficient[12] * 0.0) + (column_coef.coefficient[13] * 1.0) + (column_coef.coefficient[14] * 0.0) + (column_coef.coefficient[15] * 0.0) + (column_coef.coefficient[16] * 1.0) + (column_coef.coefficient[17] * 0.0) + (column_coef.coefficient[18] * 1.0) + (column_coef.coefficient[19] * 0.0) + (column_coef.coefficient[20] * 0.0) + (column_coef.coefficient[21] * 1.0) + (column_coef.coefficient[22] * 0.0) + (column_coef.coefficient[23] * 0.0) + (column_coef.coefficient[24] * 1.0) + (column_coef.coefficient[25] * 0.0) +  (column_coef.coefficient[26] * 0.0) + (column_coef.coefficient[27] * 1.0) + (column_coef.coefficient[28] * 1.0) + (column_coef.coefficient[29] * 0.0) + (column_coef.coefficient[30] * 0.0) + (column_coef.coefficient[31] * 0.0) + (column_coef.coefficient[32] * 0.0) + (column_coef.coefficient[33] * 1.0) + (column_coef.coefficient[34] * 0.0) + (column_coef.coefficient[35] * 0.0) + (column_coef.coefficient[36] * 0.0) + (column_coef.coefficient[37] * 0.0) + (column_coef.coefficient[38] * 0.0) + (column_coef.coefficient[39] * 1.0)


# In[ ]:


# add the sigmoid function to make the decision
# One way to make the loss function
def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))


print(sigmoid(weights_int_bias))


# In[ ]:


# according to wikipedia implementation: sigmoid function
1 / (1 + np.exp(-weights_int_bias))


# Get a single row of features and add it onto the model above and confirm if you get the same result as above.

# In[ ]:


clf_lr1.predict(x_test[:1]) # has chronic kidney disease for the 147th id


# Conclusion:
#     
# I think I was onto something because if you review the feature importance the **hemoglobin** and **specific gravity** were the most important features. I've done some tests with a uri strip to figure out if someone has an issue with their kidney and these are the parameters that point out dysfunction of the kidney. Others like having diabetes mellitus though this is related to pancreatic beta cells issues, sodium levels taken up again by the kidney and age could also be indicators. The decision tree is also interesting but could use some tuning. The intercept says that on default the patient doesn't have chronic disease as well. In future I will update the plots to show how the decision was made better. Otherwise, I'd discuss these results with a medical practitioner like a urologist. What do you think?

# In[ ]:




