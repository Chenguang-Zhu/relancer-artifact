#!/usr/bin/env python
# coding: utf-8

# # Data explolation and classification about home equity loan by using ML with random forest and naive bayes classifier  #
# 
# This project is about exploring the data,hypothesis test and classification about home equity loan, which is a loan where the obligor uses the equity of his or her home as the underlying collateral.To do this, I use the data set in which there are 5960 raws and 13 columns. For this project, I would like to answer four questions below. 
# 
# What kind of relations exist between variables in this data set?
# How two group of being undefault and default differ ?
# Is the relationship between job category and default independent ?
# What factors affect being default greatly? and what are criterion to predict whether new clients who want to use equity loan could be default or not.
# By answering these four questions, it would be very helpful for a bank or some institutions that issue home equity loan to automate the process of issuing home equity loan and know whether obligers that can not pay the loan back and factors that affect being default. Because of this reason, it is worth of doing the project and it is very meanigful to answer these four questions above.
# 
# **About the data set**
# 
# The data that I would like to use is about a home equity loan. The data can be downloaded in a form of CSV from this website https://www.kaggle.com/ajay1735/hmeq-data. The size of data is 5960 raws and 13 columns. For this data set, time period is not mentioned from kaggle and the orginal website that the data comes from 
# 
# The data set has the following characteristics: 
# 
# ● BAD: 1 = client defaulted on loan 0 = loan repaid 
# 
# ● LOAN: Amount of the loan request 
# 
# ● MORTDUE: Amount due on existing mortgage 
# 
# ● VALUE: Value of current property 
# 
# ● REASON: DebtCon = debt consolidation; HomeImp = home improvement 
# 
# ● JOB: Occupational categories 
# 
# ● YOJ: Years at present job 
# 
# ● DEROG: Number of major derogatory reports 
# 
# ● DELINQ: Number of delinquent credit lines 
# 
# ● CLAGE: Age of oldest credit line in months 
# 
# ● NINQ: Number of recent credit inquiries 
# 
# ● CLNO: Number of credit lines 
# 
# ● DEBTINC: Debt-to-income ratio 
# 
# All of unit for money is $**
# 
# 
# **Cleaning the data set**
# 
# There are many raws having missing value. Therefore, I drop the data that have missing value. The original data set is 5960 raws. After droping the data with missing value, the data size is 3364. The method that I use to do it is dropna. The codes is shows below.

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm


# In[ ]:


data = pd.read_csv("../../../input/ajay1735_hmeq-data/hmeq.csv")


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.isna().sum()


# In[ ]:


df = data.dropna()


# This is the number of NaN value and trying to drop the value with NaN**

# In[ ]:


df.shape


# This is the number of raw and columns after droping

# ## Question 1 ##
# 
# **The relation between variables**
# 
# In question 1, we try to understand the relation between variables. The major question in question 1 is "What kind of relations exist between variables in this data set?" I would like to use heat map and pair plots to answer the first question.
# 
# To do this analysis, I drop the categorical variables such as REASON,JOB and BAD and make "df_con_bad" from the data frame "df",which is the one droping the raws with missing value above. In addition, to do anaylysis of pairplot, I create df_pair,which is composed of LOAN,BAD,MORTDUE,VALUE,YOJ from "df". This is because we can not create heat map and pairplots well witout making new data frame,which is challenge that I was faced with.

# In[ ]:


df_con_bad  = df.drop("REASON",axis = 1 )
df_con_bad = df_con_bad.drop("JOB",axis = 1)


# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
df_con_bad_corr = df_con_bad.drop("BAD",axis = 1)  
corr = df_con_bad_corr.corr()
print()


# From this result, VALUE(value of home) has a strong correlation to MORTDUE(amount due on existing mortgage). This does make sence because if there is high value of home, it is likely to be high amount due on exsiting morgage. In additon, low correlation betwen Value an LOAN(amount of the loan requested) shows the characteristic of the home equity loan. Home equity loan is used as a collateral of value of home. If we have high value of home, the owners of the home could issue high amount of home equity loan. However, the amount of money requested has low correlation to Value of home,which means that the owners of high value of home does not need the home equity loan and people who suffer from money are likely to become obligoer of home equity loan.

# In[ ]:


dic = {"LOAN":df["LOAN"],"BAD":df["BAD"],"MORTDUE":df["MORTDUE"],"VALUE":df["VALUE"],"YOJ":df["YOJ"]}
rcParams['figure.figsize'] = 5, 5
df_pair = pd.DataFrame(dic)
print()


# By looking at the relation of each pairs, if one variable in pair has excessed value, it is likely to be default. For example, in the pair of LOAN and YOJ, high number of YOJ and low number of LOAN is likely to lead to being default. In the same way, in the pair of LOAN and YOJ, low number of YOJ and high number of LOAN is likely to lead to being default.

# ## Question 2 ##
# 
# **The relation between group of being default and default**
# 
# In question 2, I try to figure out the relation between group of being default and default. Main question in question 2 is "How two group of being undefault and default differ ?" To answer this question, boxplot, bar graphs, displaying summary of stats data can be used.
# 
# At first, to do make analysis in this question, I make df_set_bad by setting BAD as index. In addition, to make a bar graph, I create data frame df_set_bad_mean_undefault and df_set_bad_mean_default to extract the mean value of LOAN,MORTDUE and VALUE in two groups of being default and undefault. In addition, to creat a bar plot, I use data frame "df", which is the one that I created firstly.

# In[ ]:


df[df["BAD"]==0].shape


# The number of the data in case of undefault

# In[ ]:


df[df["BAD"]==1].shape


# The number of the data in case of default

# In[ ]:


df_set_bad = df.set_index("BAD")


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
rcParams['figure.figsize'] = 8, 5
# data to plot
n_groups = 3
df_set_bad_mean_undefault = df_set_bad.loc[0].mean()
df_set_bad_mean_default = df_set_bad.loc[1].mean()

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, df_set_bad_mean_undefault[0:3], bar_width,alpha=opacity,color='b',label='Undefault')
 
rects2 = plt.bar(index + bar_width, df_set_bad_mean_default[0:3], bar_width,alpha=opacity,color='g',label='Default')
 
plt.xlabel('Mean of each variables')
plt.ylabel('$ value')
plt.title('Comparison of mean of being undefault and default based on LOAN,MORTDUE and VALUE')
plt.xticks(index + bar_width, ("LOAN","MORTDUE","VALUE","YOJ","DEROG","DELINQ","CLAGE","NINQ","CLNO","DEBTINC"))
plt.legend()
 
plt.tight_layout()
print()


# If we compare the group of being default and undefault based on LOAN (amount of the loan requested),MORTDUE(amount due on existing mortgage) and VALUE(value of current property), there are not big differnce between them. Therefore, it is hard to predict whether the obligoer can pay the loan back based on the amount of the loan requested, amount of due on existing mortgage and value of current property.

# In[ ]:


df_set_bad.loc[0].mean()


# This is the summary of each variable for those who can pay home equity loan back

# In[ ]:


df_set_bad.loc[1].mean()


# This is the summary of each variable for those who can not pay home equity loan back
# 
# If we compare the mean of DELOG(Number of major derogatory reports),DELINQ(Number of delinquent credit lines) and DEBTINC(Debt to income ratio)between two groups of being default and undefault, there is a significant difference of the value. The mean value of DEROG for default group is about 6 times as high as the one for undefault group. In addition, the mean value of DELINQ for default group is about 4 times higher than the one for undefault group. Besides, the group of being default has higher debt to income ratio by 6% than the one of being undefault.

# In[ ]:


#pad=0.3, w_pad=4, h_pad=1.0
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 30, 15
fig, axs = plt.subplots(3,3)
plt.tight_layout()
fig.subplots_adjust(top=0.88)
sns.boxplot(x="BAD", y="LOAN", data=df,ax=axs[0,0])
axs[0,0].set_title(" Amount of the loan requesed for a group of being default and undefault\n'",fontsize=14)
sns.boxplot(x="BAD", y="MORTDUE", data=df,ax=axs[0,1])
axs[0,1].set_title(" Amount due on existing mortgage for a group of being default and undefault\n'",fontsize=14)
sns.boxplot(x="BAD", y="VALUE", data=df,ax=axs[0,2])
axs[0,2].set_title(" Value of current property for a group of being default and undefault\n'",fontsize=14)

sns.boxplot(x="BAD", y="YOJ", data=df,ax=axs[1,0])
axs[1,0].set_title("Years at present job for a group of being default and undefault\n'",fontsize=14)

sns.boxplot(x="BAD", y="DEROG", data=df,ax=axs[1,1])
axs[1,1].set_title(" Number of major derogatory report for a group of being default and undefault\n'",fontsize=14)

sns.boxplot(x="BAD", y="DELINQ", data=df,ax=axs[1,2])
axs[1,2].set_title("     Number of delinquent credit lines  for a group of being default and undefault\n'",fontsize=14)

sns.boxplot(x="BAD", y="CLAGE", data=df,ax=axs[2,0])
axs[2,0].set_title(" Age of oldest credit line in months for a group of being default and undefault\n'",fontsize=14)


sns.boxplot(x="BAD", y="CLNO", data=df,ax=axs[2,1])
axs[2,1].set_title(" Number of credit lines for a group of being default and undefault\n'",fontsize=14)


sns.boxplot(x="BAD", y="DEBTINC", data=df,ax=axs[2,2])
axs[2,2].set_title("Debt-to-income ratio for a group of being default and undefault\n'",fontsize=14)

plt.tight_layout()
print()


# If we compare two groups of being default and undefault, most of them has no big difference between them. However, the variables such as DEROG and DELINQ has difference betwee them. People who can not pay the loan back are likely to have higher number of derogatory reports.In addition, The high number of deliquent credit lines tend to lead to the obligor being default. Based on this result, if there are high number of deliquent credit lines and derogatory reports, banks or some institution who are lending hoem equity loan should keep an eye on these obligor carefully and take some measurement in order not to lose the money that they are lending.

# # Question 3 #
# 
# **The relation between job categories and the grops of being defaul and undefault**
# 
# In question 3, I would like to figure out the relation between job category and default. The main queston in this question is about "Is the relationship between job category and default independent ?" I use bar plots and chi square test to answer this question.
# 
# Firstly, by setting JOB as index, I make df_set_job. Through this data frame, I show the rate of default based on each job. To know about the difference of value of variables in each job positon, bar plot is created. In addition,in order to do chi square test, I made a cross table and do hypothesis test about chi square.
# 
# 

# In[ ]:


df_set_job = df.set_index("JOB")


# In[ ]:


df_set_job["BAD"].value_counts()


# In[ ]:


import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

fig, axs = plt.subplots(2,4)
plt.tight_layout(pad=0.5, w_pad=4, h_pad=1.0)

sns.boxplot(x="JOB", y="LOAN", data=df,ax=axs[0,0])
sns.boxplot(x="JOB", y="MORTDUE", data=df,ax=axs[0,1])
sns.boxplot(x="JOB", y="VALUE", data=df,ax=axs[0,2])
sns.boxplot(x="JOB", y="YOJ", data=df,ax=axs[0,3])
sns.boxplot(x="JOB", y="CLAGE", data=df,ax=axs[1,0])
sns.boxplot(x="JOB", y="NINQ", data=df,ax=axs[1,1])
sns.boxplot(x="JOB", y="CLNO", data=df,ax=axs[1,2])


# According to the graph, people that mange their own company tend to have larger amount of loan request,amount of due on exsiting mortgage and value of current property than other jobs. In addition, By looking at bar plot of CLAGE and JOB, range of self is very small. This shows that age of oldest credit line in months is likely to be small and people who mange their own company tend to able to pay the equity loan back very shortly
# 
# **Hypothesis test: Two groups of being undefault or default and about job position are independent or not ?** 
# 
# In order to test it, we use chi square test. In this hypothesis, I use Alpha level 5% and 2-tailed test. At first, I make cross table for this.

# In[ ]:


ct = pd.crosstab(df.BAD,df.JOB,margins=True) #making cross table  
ct


# This is the null hypothesis and alternative hypothesis to answer the question above.
# 
# Null Hypothesis -> two group of being undefault or default and job position of peple who have borrow home equity loan are independent 
# 
# Alternative Hypothesis -> two group of being undefault or default and about job position are not independent 
# 
# By answering the question, we could figure out whether job positon of the obliger affects the results of being default or not.

# In[ ]:


from scipy.stats import chi2_contingency
chi2, p, dof, ex = chi2_contingency(ct)
print("chi2 = ", chi2)
print("p-val = ", p)
print("degree of freedom = ",dof)
print("Expected:")
pd.DataFrame(ex)


# Since p value is 0.00029481726266075394, we can reject the null hypothesis,which means that two group of being undefault or default and about job position are not independent. It implies that job positions affect whether the obligoer return its home equity loan or not.

# ## Question 4 ##
# 
# **Prediction through clasification methods such as decetion tree, random forest and Naive Bayes Classifier**
# 
# In question 4, I would like to predict being default or not based on clasification models such as decision tree, random forest and bayes classifiers. The main theme in this question is "what factors affect default greatly? and what are criterion to predict whether new clients who want to use equity loan could be default or not.
# 
# 
# To do classification, I split 70 % of the data into train data and 30% of the data into test data. In addition, new data frames df_train_drop_cate and df_test_drop_cate that drop categorical variables are created. By using these data frame, classification models such as Desition tree, random forest and Bayes Naives

# In[ ]:


import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import sklearn.ensemble as skens
import sklearn.metrics as skmetric
import sklearn.naive_bayes as sknb
import sklearn.tree as sktree
import matplotlib.pyplot as plt
print()
import seaborn as sns
sns.set(style='white', color_codes=True, font_scale=1.3)
import sklearn.externals.six as sksix
import IPython.display as ipd
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import os


# **Desition tree**
# 
# Objective: predicting whether the obligor can pay home equity loan or nor based on the several features. 
# Possible classes: BAD=0(the obligor can pay the home equity loan) or BAD = 1 (the obligor can not pay the home equity loan) 
# Features: all features except of categorial varibales such as JOB,BAD and REASON

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df_train,df_test = train_test_split(df, test_size=0.3)
df_train_drop_cate = df_train.drop("REASON",axis=1) #droping the categorical variables 
df_test_drop_cate = df_test.drop("REASON",axis=1)
df_train_drop_cate = df_train_drop_cate.drop("JOB",axis=1)
df_test_drop_cate = df_test_drop_cate.drop("JOB",axis=1)
df_train_drop_cate = df_train_drop_cate.drop("BAD",axis=1)
df_test_drop_cate = df_test_drop_cate.drop("BAD",axis=1)


# In[ ]:


df_train.shape


# In[ ]:


s = df_train_drop_cate


# In[ ]:


df_train_drop_cate_show = s
df_train_drop_cate_show["BAD"] = df_train["BAD"]


# In[ ]:


df_train_drop_cate = df_train_drop_cate.drop("BAD",axis=1)


# In[ ]:


dt_model = sktree.DecisionTreeClassifier(max_depth=3, criterion='entropy') 
dt_model.fit(df_train_drop_cate,df_train_drop_cate_show.BAD)


# In[ ]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import graphviz

export_graphviz(dt_model, out_file="tree_dt_model.dot", filled=True, rounded=True, special_characters=True,feature_names=df_train_drop_cate.columns) 
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#Image(filename = 'tree_dt_model.dot')


# In[ ]:


print()


# In[ ]:


from IPython.display import Image
Image(filename = 'tree_dt_model.png')


# In[ ]:


predicted_labels = dt_model.predict(df_train_drop_cate)
df_train_drop_cate_show['predicted_dt_tree'] = predicted_labels


# This is the desiton tree based on the data about the home equity loan. It shows how being default and undefault are clasified and what factors affect the clasification of being default. The first important factor is Debt to income ratio. When "DEBTINC(bebt to income ratio) is less than or equal to 43.68" for a given sample, that sample is assigned to the bottom left node. When it is false, the sample is assigned to the bottom right node. It keeps doing for each node of criterio to clasify samples.Besides, for this desion tree, the color of orange means that it is clasified as being undefault while that of the blue shows that the sample is done as being default. The pale color of blue and orange means entropy,which is the quantification of uncertainty in data. High entropy means higg uncertainy about predicting the result of being default or not.

# In[ ]:


len(df_train_drop_cate_show[df_train_drop_cate_show['BAD'] == df_train_drop_cate_show["predicted_dt_tree"]])/len(df_train_drop_cate_show)


# This is the accuracy rate based on the model of the desition tree above. This implies that based on the criterio of each node above the graph of detision tree, banks or some insitution are prone to predict whether the obligor can pay the home equity loan back to them at 93%.

# In[ ]:


def comparePlot(input_frame,real_column,predicted_column):
    df_a = input_frame.copy()
    df_b = input_frame.copy()
    df_a['label_source'] = 'BAD'
    df_b['label_source'] = 'Classifier'
    df_a['label'] = df_a[real_column]
    df_b['label'] = df_b[predicted_column].apply(lambda x: 'Predict %s'%x)
    df_c = pd.concat((df_a, df_b), axis=0, ignore_index=True)
    sns.lmplot(x='DEBTINC', y='CLAGE', col='label_source', hue='label', data=df_c, fit_reg=False, size=3); 


# In[ ]:


comparePlot(df_train_drop_cate_show,"BAD","predicted_dt_tree")


# This is the graphs of the plot of observed data and prediceted data. Most of plots on the right side and left side are matched wit each other.

# **Random Forest**
# 
# Objective 
# predicting whether the obligor can pay home equity loan or nor based on the several features.
# 
# Possible classes 
# BAD=0(the obligor can pay the home equity loan) or BAD = 1 (the obligor can not pay the home equity loan)
# 
# Features 
# all features except of categorial varibales such as JOB,BAD and REASON

# In[ ]:


rf_model = skens.RandomForestClassifier(n_estimators=10,oob_score=True, criterion='entropy')
rf_model.fit(df_train_drop_cate,df_train.BAD)


# This is the model for random forest

# In[ ]:


feat_importance = rf_model.feature_importances_
pd.DataFrame({'Feature Importance':feat_importance}, index=df_train_drop_cate.columns).plot(kind='barh') 
rcParams['figure.figsize'] = 10, 5


# From this result, DEBTINC(ratio of debt to income) is the most important variable among them. It shows that if the ratio of debt to income is high, the obligoer can not pay home equity loan back. In addition, CLAGE is the second important variable. This means that having other loans longer tends to lead to being default.

# In[ ]:


predicted_labels = rf_model.predict(df_train_drop_cate)
df_train_drop_cate_show['predicted_rf_tree'] = predicted_labels


# In[ ]:


comparePlot(df_train_drop_cate_show,"BAD","predicted_rf_tree")


# According to the graph above, most of them are predcted by the random forest.
# 
# Then, I would like to do the cross varidation to see the best model for random forest.

# In[ ]:


param_grid = { 'n_estimators': [5, 10, 15, 20, 25], 'max_depth': [2, 5, 7, 9], } 


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


grid_clf = GridSearchCV(rf_model, param_grid, cv=10)
grid_clf.fit(df_train_drop_cate,df_train.BAD)


# In[ ]:


grid_clf.best_estimator_


# In[ ]:


grid_clf.best_params_ 


# In[ ]:


#Turing the model into the one with max_depth': 9, 'n_estimators': 20
rf_model2 = skens.RandomForestClassifier(n_estimators=20,oob_score=True,max_depth=9, criterion='entropy')
rf_model2.fit(df_train_drop_cate,df_train.BAD)
predicted_labels2 = rf_model.predict(df_train_drop_cate)
df_train_drop_cate_show['predicted_rf_tree2'] = predicted_labels2


# In[ ]:


len(df_train_drop_cate_show[df_train_drop_cate_show['BAD'] == df_train_drop_cate_show["predicted_rf_tree2"]])/len(df_train_drop_cate_show)


# This is the acuuracy rate of 99%, which is very high to predict about being default and undefault

# # Naive Bayes Classifier #
# 
# Two possible classes: "default" and "undefault". 
# Features:CLAGE(age of oldest credit lines) and DEBTINC(Debt to income ratio)

# In[ ]:


df["is_default"] = np.where(df["BAD"] == 1,"default","not_default")
df_bay_train,df_bay_test = train_test_split(df, test_size=0.3)


# In[ ]:



gnb_model = sknb.GaussianNB()

gnb_model.fit(df_bay_train[['DEBTINC']],df_bay_train['is_default'])


# In[ ]:


# test the model
y_pred = gnb_model.predict(df_bay_test[['DEBTINC']])
df_bay_test['predicted_nb'] = y_pred


# In[ ]:


comparePlot(df_bay_test,"is_default","predicted_nb")


# In[ ]:


len(df_bay_test[df_bay_test['is_default'] == df_bay_test["predicted_nb"]])/len(df_bay_test)


# This is the acuuracy rate of precition with Naive Bayes Classifier

# ## Conclusion ##
# 
# **Question 1: What kind of relations exist between variables in this data set?**
# 
# There are not many high correlation between varianbles except for VALUE(vale of current property) & MORTDUE(amount due on existing mortgage),which is very normal because if there is high value of home, it is likely to be high amount due on exsiting morgage. In addition, low correlation betwen VALUE an LOAN(amount of the loan requested) shows the characteristic of the home equity loan in which the owners of high value of home does not need the home equity loan and people who suffer from money are likely to become obligoer of home equity loan. Besides, among pairs of variables such as LOAN,VALUE, YOJ, if one variable in pair has excessed value, the obligoers of the home equity loan are likely to be default.
# 
# **Question 2 : How two group of being undefault and default differ ?**
# 
# In comparison of the two groups of being default and undefault based on the numerical variables, there are not big differnce between LOAN (amount of the loan requested),MORTDUE(amount due on existing mortgage) and VALUE(value of current property).However, there is a significant difference of DELOG(Number of major derogatory reports),DELINQ(Number of delinquent credit lines) and DEBTINC(Debt to income ratio)between two groups of being default and undefault. In fact, the mean value of DEROG for default group is about 6 times as high as the one for undefault group. In addition, the mean value of DELINQ for default group is about 4 times higher than the one for undefault group. Besides,the group of being default has higher debt to income ratio by 6% than the one of being undefault. This means that people who can not return the loan back are prone to have higher number of derogatory reports. In addition, The high number of deliquent credit lines tend to lead to the obligor being default. Based on this result, if there are high number of deliquent credit lines and derogatory reports, banks or some institution who are lending hoem equity loan should keep an eye on these obligor carefully and take some measurement in order to prevent them from losiing the money that they are ledning
# 
# **Question 3 : Is the relationship between job category and default independent ?** 
# 
# The relationship between job category and default independent based on the result of chi square test. This means that what job the obligoer have affect whether they are able to pay the money of home equity loan back or not. In addtion, people that mange their own company tend to have larger amount of loan request,amount of due on exsiting mortgage and value of current property than other jobs. Besides, by looking at bar plot of CLAGE and JOB, range of self is very small. This shows that people who mange their own company tend to able to pay the equity loan back very shortly
# 
# **Question 4 :What factors affect being default greatly? and what are criterion to predict whether new clients who want to use equity loan could be default or not.** 
# 
# According the results from decition tree,random forest and Naive Bayes classifer, the highest accuracy rate is the one with random forest among three methods, which is 99% to predict about being default or undefault. From the feature importance of random forest, DEBTINC(Debt income ratio) and CLAGE(Age of oldest credit line in months) are important to classfy the samples. In addition, desition tree is also helpful to support the interpretation of clasification. Although the acuracty rate of clasification of desition tree is lower than random forest, the desition tree shows us criterions to clasify samples and future data. Based on the visualzied desition tree above, banks or some institutions issuing home equity loan could automate the process of issuing home equity loan and predict the futre possible clients could be default or not at 93% accuracy.

# In[ ]:




