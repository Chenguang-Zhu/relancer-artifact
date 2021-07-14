#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-info">
# <h1><center><font color=darkblue> HEART DISEASE PREDICTION USING LOGISTIC REGRESSION.<font></center></h1>
# 
# 
# </div>

# ## <font color=RoyalBlue>Introduction<font>
# 
# World Health Organization has  estimated 12 million deaths occur worldwide, every year due to Heart diseases. Half the deaths in the United States and other developed countries are due to cardio vascular diseases. The early prognosis of cardiovascular diseases can aid in making decisions on lifestyle changes in high risk patients and in turn reduce the complications. This research intends to pinpoint the most relevant/risk factors of heart disease as well as predict the overall risk using logistic regression.
# 

# In[16]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
print()


# ## <font color=RoyalBlue>Data Preparation<font>
# 
# ### <font color=CornflowerBlue>Source:<font>
# 
# The dataset is publically available on the Kaggle website, and it is from an ongoing ongoing cardiovascular study on residents of the town of Framingham, Massachusetts.  The classification goal is to predict whether the patient has 10-year risk of future coronary heart disease (CHD).The dataset provides the patients’ information. It includes over 4,000 records and 15 attributes.

# In[19]:


heart_df=pd.read_csv("../../../input/amanajmera1_framingham-heart-study-dataset/framingham.csv")
heart_df.drop(['education'],axis=1,inplace=True)
heart_df.head()


# ### <font color=CornflowerBlue>Variables :<font>

# Each attribute is a potential risk factor. There are both demographic, behavioural and medical risk factors.
# 
#  - **<font color=SteelBlue>Demographic:<font>**
# sex: male or female;(Nominal)
# 
#     -  age: age of the patient;(Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous)
# 
# 
#  - **<font color=SteelBlue>Behavioural<font>**
# 
#     -  currentSmoker: whether or not the patient is a current smoker (Nominal)
# 
#     -  cigsPerDay: the number of cigarettes that the person smoked on average in one day.(can be considered continuous as one can have any number of cigarretts, even half a cigarette.)
# 
#  - **<font color=SteelBlue>Medical( history):<font>**
# 
#     -  BPMeds: whether or not the patient was on blood pressure medication (Nominal)
# 
#     -  prevalentStroke: whether or not the patient had previously had a stroke (Nominal)
# 
#     -  prevalentHyp: whether or not the patient was hypertensive (Nominal)
# 
#     -  diabetes: whether or not the patient had diabetes (Nominal)
# 
#  - **<font color=SteelBlue>Medical(current):<font>** 
# 
#     -  totChol: total cholesterol level (Continuous)
# 
#     -  sysBP: systolic blood pressure (Continuous)
# 
#     -  diaBP: diastolic blood pressure (Continuous)
# 
#     -  BMI: Body Mass Index (Continuous)
# 
#     -  heartRate: heart rate (Continuous - In medical research, variables such as heart rate though in fact discrete, yet are considered continuous because of large number of possible values.)
# 
#     -  glucose: glucose level (Continuous)
# 
# 
#  - **<font color=SteelBlue>Predict variable (desired target):<font>**
# 
#     -  10 year risk of coronary heart disease CHD (binary: “1”, means “Yes”, “0” means “No”)
# 

# In[20]:


heart_df.rename(columns={'male':'Sex_male'},inplace=True)


# ### <font color=CornflowerBlue>Missing values<font>

# In[21]:


heart_df.isnull().sum()


# In[22]:


count=0
for i in heart_df.isnull().sum(axis=1):
    if i>0:
        count=count+1
print('Total number of rows with missing values is ', count)
print('since it is only',round((count/len(heart_df.index))*100), 'percent of the entire dataset the rows with missing values are excluded.')


# In[23]:


heart_df.dropna(axis=0,inplace=True)


# ## <font color=RoyalBlue>Exploratory Analysis<font>

# In[6]:


def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+" Distribution",color='DarkRed')
        
    fig.tight_layout()  
    print()
draw_histograms(heart_df,heart_df.columns,6,3)


# In[7]:


heart_df.TenYearCHD.value_counts()


# In[8]:


sn.countplot(x='TenYearCHD',data=heart_df)


# There are 3179 patents with no heart disease and 572 patients with risk of heart disease.

# In[24]:


print()


# In[25]:


heart_df.describe()


# ## <font color=RoyalBlue>Logistic Regression<font>

# Logistic regression is a type of regression analysis in statistics used for prediction of outcome of a categorical dependent variable from a set of predictor or independent variables. In logistic regression the dependent variable is always binary. Logistic regression is mainly used to for prediction and also calculating the probability of success. 

# In[26]:


from statsmodels.tools import add_constant as add_constant
heart_df_constant = add_constant(heart_df)
heart_df_constant.head()


# In[31]:


st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)
cols=heart_df_constant.columns[:-1]
model=sm.Logit(heart_df.TenYearCHD,heart_df_constant[cols])
result=model.fit()
result.summary()


# The results above show some of the attributes with P value higher than the preferred alpha(5%) and thereby showing  low statistically significant relationship with the probability of heart disease. Backward elemination approach is used here to remove those attributes with highest Pvalue one at a time follwed by  running the regression repeatedly until all attributes have P Values less than 0.05.
# 
# 

# ### <font color=CornflowerBlue>Feature Selection: Backward elemination (P-value approach)<font>

# In[32]:


def back_feature_elem (data_frame,dep_var,col_list):
    """ Takes in the dataframe, the dependent variable and a list of column names, runs the regression repeatedly eleminating feature with the highest P-value above alpha one at a time and returns the regression summary with all p-values below alpha""" 

    while len(col_list)>0 :
        model=sm.Logit(dep_var,data_frame[col_list])
        result=model.fit(disp=0)
        largest_pvalue=round(result.pvalues,3).nlargest(1)
        if largest_pvalue[0]<(0.05):
            return result
            break
        else:
            col_list=col_list.drop(largest_pvalue.index)

result=back_feature_elem(heart_df_constant,heart_df.TenYearCHD,cols)


# In[33]:


result.summary()


# #### <font color=darkblue>Logistic regression equation<font>
# 
# $$P=\hspace{.2cm}e^{\beta_0 + \beta_1 X_1}\hspace{.2cm}/\hspace{.2cm}1+e^{\beta_0 +\beta_1 X_1}$$
# 
# When all features plugged in:
# 
# $$logit(p) = log(p/(1-p))=\beta_0 +\beta_1\hspace{.1cm} *\hspace{.2cm} Sexmale\hspace{.2cm}+\beta_2\hspace{.1cm} * \hspace{.1cm}age\hspace{.2cm}+\hspace{.2cm}\beta_3\hspace{.1cm} *\hspace{.1cm} cigsPerDay\hspace{.2cm}+\hspace{.2cm}\beta_4 \hspace{.1cm}*\hspace{.1cm} totChol\hspace{.2cm}+\hspace{.2cm}\beta_5\hspace{.1cm} *\hspace{.1cm} sysBP\hspace{.2cm}+\hspace{.2cm}\beta_6\hspace{.1cm} *\hspace{.1cm} glucose\hspace{.2cm}$$
# 
# 

# ## <font color=RoyalBlue>Interpreting the results: Odds Ratio, Confidence Intervals and Pvalues<font>

# In[34]:


params = np.exp(result.params)
conf = np.exp(result.conf_int())
conf['OR'] = params
pvalue=round(result.pvalues,3)
conf['pvalue']=pvalue
conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']
print ((conf))


#  - **This fitted model shows that, holding all other features constant, the odds of getting diagnosed with heart disease for males (sex_male = 1)over that of  females (sex_male = 0) is exp(0.5815) = 1.788687. In terms of percent change, we can say that the odds for males are 78.8% higher than the odds for females.**
# <br>
# <br>
# 
#  - **The coefficient for age says that, holding all others constant, we will see 7% increase in the odds of getting diagnosed with CDH for a one year increase in age since exp(0.0655) = 1.067644.**
# <br>
# <br>
#  - **Similarly , with every extra cigarette one smokes thers is a 2% increase in the odds of CDH.** 
# <br>
# <br>
#  - **For Total cholosterol level and glucose level there is no significant change.**
# <br>
# <br>
#  - **There is a 1.7% increase in odds for every unit increase in systolic Blood Pressure.**
# 

# ### <font color=CornflowerBlue>Splitting data to train and test split<font>

# In[35]:


import sklearn
new_features=heart_df[['age','Sex_male','cigsPerDay','totChol','sysBP','glucose','TenYearCHD']]
x=new_features.iloc[:,:-1]
y=new_features.iloc[:,-1]
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)


# In[36]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)


# ## <font color=RoyalBlue>Model Evaluation<font>
# 
# ### <font color=CornflowerBlue>Model accuracy<font>

# In[37]:


sklearn.metrics.accuracy_score(y_test,y_pred)


# ####  <font color=DarkBlue>Accuracy of the model is 0.88<font>

# ### <font color=CornflowerBlue>Confusion matrix<font>

# In[38]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
print()


# The confusion matrix shows 658+4 = 662 correct predictions and 88+1= 89 incorrect ones.
# 
# **<font color=DarkBlue>True Positives:**  4<font>
# 
# **<font color=DarkBlue>True Negatives:**  658<font>
# 
# **<font color=DarkBlue>False Positives:** 1 (*Type I error*)<font>
# 
# **<font color=DarkBlue>False Negatives:** 88 ( *Type II error*)<font>

# In[39]:


TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)


# ### <font color=CornflowerBlue>Model Evaluation - Statistics<font>

# In[40]:


print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',  'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',  'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',  'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',  'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',  'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',  'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',  'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity) 


# **From the above statistics it is clear that the model is highly specific than sensitive. The negative values are predicted more accurately than the positives.**

# ### <font color=CornflowerBlue>Predicted probabilities of  0 (No Coronary Heart Disease) and 1 ( Coronary Heart Disease: Yes)  for the test data with a default classification threshold of 0.5<font>
# 

# In[41]:


y_pred_prob=logreg.predict_proba(x_test)[:,:]
y_pred_prob_df=pd.DataFrame(data=y_pred_prob, columns=['Prob of no heart disease (0)','Prob of Heart Disease (1)'])
y_pred_prob_df.head()


# ### <font color=CornflowerBlue>Lower the threshold<font>

# Since the model is predicting Heart disease too many type II errors is not advisable. A False Negative ( ignoring the probability of disease when there actualy is one) is more dangerous than a False Positive in this case. Hence inorder to increase the sensitivity,  threshold can be lowered.

# In[42]:


from sklearn.preprocessing import binarize
for i in range(1,5):
    cm2=0
    y_pred_prob_yes=logreg.predict_proba(x_test)
    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]
    cm2=confusion_matrix(y_test,y_pred2)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n', 'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n', 'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n') 
    


# ### <font color=CornflowerBlue>ROC curve<font>

# In[43]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:,1])
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Heart disease classifier')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)


# A common way to visualize the trade-offs of different thresholds is by using an ROC curve, a plot of the true positive rate (# true positives/ total # positives) versus the false positive rate (# false positives /
# total # negatives) for all possible choices of thresholds. A model with good classification accuracy should have significantly more true positives than false positives at all thresholds. 
# 
# The optimum position for roc curve is towards the top left corner where the specificity and sensitivity are at optimum levels

# ### <font color=CornflowerBlue>Area Under The Curve (AUC)<font>
# 
# The area under the ROC curve quantifies model classification accuracy; the higher the area, the greater the disparity between true and false positives, and the stronger the model in classifying members of the training dataset. An area of 0.5 corresponds to a model that performs no better than random classification and a good classifier stays as far away from that  as possible. An area of 1 is ideal. The closer the AUC to 1 the better.

# In[44]:


sklearn.metrics.roc_auc_score(y_test,y_pred_prob_yes[:,1])


# 
# ## <font color=RoyalBlue> Conclusions:</font>
# 
# <div class="alert alert-info">
# 
# 
# 
# <div class="panel-body">
# 
# 
#  - **<font color=darkblue>All attributes selected after the elimination process show Pvalues lower than 5% and thereby suggesting significant role in the Heart disease prediction.</font>** 
# <br>
# <br>
#  - **<font color=darkblue>Men seem to be more susceptible to heart disease than women.Increase in Age,number of cigarettes smoked per day and systolic Blood Pressure also show increasing odds of having heart disease.</font>**
#  <br>
#  <br>
# 
#  - **<font color=darkblue>Total cholesterol shows no significant change in the odds of CHD. This could be due to the presence of 'good cholesterol(HDL) in the total cholesterol reading.Glucose too causes a very negligible change in odds (0.2%)</font>**
#  <br>
#  <br>
# 
#  - **<font color=darkblue>The model predicted with 0.88 accuracy. The model is more specific than sensitive.</font>**
#  <br>
#  <br>
# 
#  - **<font color=darkblue>The Area under the ROC curve is 73.5 which is somewhat satisfactory.</font> **
#  <br>
#  <br>
# 
#  - ** <font color=darkblue>Overall model could be improved with more data.</font>**
# 
# </div>
# </div>

# ## <font color=RoyalBlue>Appendix
# 
# http://www.who.int/mediacentre/factsheets/fs317/en/
# 
# #### Data Source References
# 
# https://www.kaggle.com/amanajmera1/framingham-heart-study-dataset/data
# 
# 

# In[ ]:




