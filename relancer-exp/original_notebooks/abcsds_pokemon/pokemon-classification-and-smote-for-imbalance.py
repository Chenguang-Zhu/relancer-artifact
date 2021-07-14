#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import pandas_profiling as pd_prof
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set()

import matplotlib.pyplot as plt
import os
print(os.listdir("../../../input/abcsds_pokemon"))


# In[ ]:


poke_data = pd.read_csv("../../../input/abcsds_pokemon/Pokemon.csv")


# ## Initial EDA

# In[ ]:


poke_data.info()


# #### Initially only one column seems to have null values, 'Type 2'. Going by the data description it is the second column to depict the type of pokemon so it is acceptable to be null at some places because it is not necessary that every pokemon would have more than one type.

# #### Column 'Total' is the Sum of Attack, Sp. Atk, Defense, Sp. Def, Speed and HP

# In[ ]:


poke_data.head()


# #### This # column seems to be useful as the index for this dataframe, Let's load the dataframe again with # as index

# In[ ]:


poke_data = pd.read_csv("../../../input/abcsds_pokemon/Pokemon.csv",index_col='#')


# In[ ]:


poke_data.head()


# #### Let's do a summary statistic analysis for the data

# In[ ]:


poke_data.describe()


# In[ ]:


#pd_prof.ProfileReport(poke_data)


# #### We should replace the bool values in Legendary column with binary values

# In[ ]:


poke_data.Legendary.replace({True:1,False:0},inplace=True)


# #### Let's check now

# In[ ]:


poke_data.describe()


# #### No doubt the data would require to be scaled before modelling can take place. The max values of all columns are varying widely.
# #### Apart from that none of the statistics seem to be unrealistic like negative values etc. so lets move forward.

# In[ ]:


p = poke_data.hist(figsize = (20,20))


# #### Some of the factors like Sp. Atk, Defence seem to be skewed.

# ## Skewness
# 
# A ***left-skewed distribution*** has a long left tail. Left-skewed distributions are also called negatively-skewed distributions. That’s because there is a long tail in the negative direction on the number line. The mean is also to the left of the peak.
# 
# A ***right-skewed distribution*** has a long right tail. Right-skewed distributions are also called positive-skew distributions. That’s because there is a long tail in the positive direction on the number line. The mean is also to the right of the peak.
# 
# 
# ![](https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2014/02/pearson-mode-skewness.jpg)
# 
# 
# #### to learn more about skewness
# https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/skewed-distribution/

# ### Unique Values in each column

# In[ ]:


poke_data['Legendary'].value_counts()


# #### Great class imbalance!

# In[ ]:


poke_data['Type 1'].value_counts()


# In[ ]:


poke_data['Type 2'].value_counts()


# #### Two columns have the same categories. This may cause issues with one hot encoding. Let's try to solve for this.
# #### Beginning by checking what values the two columns contain and are they same.

# In[ ]:


type_1_list = list(poke_data['Type 1'].value_counts().index)
type_2_list = list(poke_data['Type 2'].value_counts().index)


# In[ ]:


type_1_list.sort()==type_2_list.sort()


# #### This shows that the values in both the type columns are same. That means no column has a different category.

# ### If get_dummies is directly applied to change this categorical data to binary, the function will produce columns with the same name because of the presence of same categories in both columns and due to the nature of get_dummies function. This would cause high dimensionality and unnecessary redundancy.
# #### To deal with this problem dummies of the two columns are created separately.

# In[ ]:


dummy_type_1 = pd.get_dummies(poke_data['Type 1'])
dummy_type_2 = pd.get_dummies(poke_data['Type 2'])


# #### Now iterating on the name of the categories present in one of the columns (any one because both columns have same categories), the values of the columns having same name would be added together and stored in a third dataframe with the same index as the initial dataframe.

# In[ ]:


dummy_final = pd.DataFrame(index=poke_data.index)
for column_name in type_2_list:
    dummy_final[column_name] = dummy_type_1[column_name] + dummy_type_2[column_name]


# In[ ]:


dummy_final.head()


# #### To check whether the steps we took are correct, summary stats are printed for the new dataframe. None of the columns contains a max value greater than one. Hence there's nothing to worry about.

# In[ ]:


dummy_final.describe()


# In[ ]:


dummy_final.info()


# #### By printing the info we can see that this new dataframe doesn't contain any Nan values whereas we had seen that the type 2 column had Nans. So what just happened? Where did the Nan values go?
# 
# #### Let's try to understand using a very raw example that I tried to depict below. 

# In[ ]:


# I have a dataframe 'df' like this 

# Id    v1    v2
# 0     A     0.23
# 1     B     0.65
# 2     NaN   0.87

# If I use this function

# df1 = get_dummies(df)
# df1

# Id    v1_A    v1_B    v2
# 0     1       0       0.23
# 1     0       1       0.65
# 2     0       0       0.87 .


# #### So it is visible that get_dummies function converts Nans to 0 and doesn't form a separate column for them like it does for other categories. Our problem has been solved thanks to get_dummies()

# #### Now we can concatenate this dataframe with the initial dataframe and drop the type 1 and type 2 columns

# In[ ]:


poke_data_new = pd.concat([poke_data,dummy_final],sort=False,axis=1)


# #### Now let's do a basic check on this!

# In[ ]:


poke_data_new.head()


# In[ ]:


poke_data_new.drop(['Type 1','Type 2'],axis=1,inplace=True)
poke_data_new.info()


# #### Perfect we'll be using this one from now. 

# #### Finally let's see the column Generation

# In[ ]:


poke_data_new['Generation'].value_counts()


# ## Bivariate EDA

# In[ ]:


plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(poke_data.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap


# #### Noticable that none of the factors seem to have a correlation with the target value higher than 0.70

# In[ ]:


poke_data_new.columns


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
numerical =  pd.DataFrame(sc_X.fit_transform(poke_data_new[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def','Speed']]),columns=['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def','Speed'],index= poke_data_new.index)


# In[ ]:


#numerical
poke_clean_standard = poke_data_new.copy(deep=True)
poke_clean_standard[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def','Speed']] = numerical[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def','Speed']]


# In[ ]:


poke_clean_standard.head()


# In[ ]:


poke_clean_standard.describe()


# In[ ]:


x = poke_clean_standard.drop(["Legendary","Name"],axis=1)
y = poke_clean_standard.Legendary


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y,random_state = 2,test_size=0.4,stratify=y)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))


# In[ ]:


## score that comes from testing on the same datapoints that were used for training
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))


# In[ ]:


## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))


# ## Result visualisation

# In[ ]:


plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')


# #### The best result is captured at k = 7 hence 7 is used for the final model 

# In[ ]:


#Setup a knn classifier with k neighbors
#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(7)

knn.fit(X_train,y_train)
knn.score(X_test,y_test)


# ## Confusion Matrix

# In[ ]:


y_pred = knn.predict(X_test)


# In[ ]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# # F1 Score

# In[ ]:


from sklearn.metrics import f1_score
f1_score(y_test,y_pred)


# # Classification Report

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# # Matthew Correlation Coefficient Score

# In[ ]:


from sklearn.metrics import matthews_corrcoef
print(matthews_corrcoef(y_test,y_pred))


# ## ROC Curve

# In[ ]:


from sklearn.metrics import roc_curve
y_pred_proba = knn.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=7) ROC curve')
print()


# # SMOTE for Class Imbalance

# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


poke_clean_standard.Legendary.value_counts()


# In[ ]:


sm = SMOTE(random_state=2, ratio = 'minority')
x_train_res, y_train_res = sm.fit_sample(X_train, y_train)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(x_train_res,y_train_res)
    
    train_scores.append(knn.score(x_train_res,y_train_res))
    test_scores.append(knn.score(X_test,y_test))


# In[ ]:


## score that comes from testing on the same datapoints that were used for training
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))


# In[ ]:


## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))


# In[ ]:


knn = KNeighborsClassifier(2)

knn.fit(x_train_res,y_train_res)
knn.score(X_test,y_test)


# In[ ]:


y_pred = knn.predict(X_test)


# In[ ]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[ ]:


from sklearn.metrics import f1_score
f1_score(y_test,y_pred)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:


from sklearn.metrics import matthews_corrcoef
print(matthews_corrcoef(y_test,y_pred))


# # SMOTE Tomek Method

# In[ ]:


from imblearn.combine import SMOTETomek

smt = SMOTETomek(ratio='auto')
x_train_res, y_train_res = smt.fit_sample(X_train, y_train)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(x_train_res,y_train_res)
    
    train_scores.append(knn.score(x_train_res,y_train_res))
    test_scores.append(knn.score(X_test,y_test))


# In[ ]:


## score that comes from testing on the same datapoints that were used for training
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))


# In[ ]:


## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))


# In[ ]:


knn = KNeighborsClassifier(2)

knn.fit(x_train_res,y_train_res)
knn.score(X_test,y_test)


# In[ ]:


y_pred = knn.predict(X_test)


# In[ ]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[ ]:


from sklearn.metrics import f1_score
f1_score(y_test,y_pred)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:


from sklearn.metrics import matthews_corrcoef
print(matthews_corrcoef(y_test,y_pred))


# # Yet To Be Updated

# In[ ]:





