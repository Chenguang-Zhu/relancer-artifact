#!/usr/bin/env python
# coding: utf-8

# # CONTENT
# 1. [Introduction](#1)
# 2. [Load and Check Data](#2)
# 3. [Outlier Detection](#3)
# 4. [Fill Missing Value](#4)
# 5. [Data Visualization](#5)
# 6. [Machine Learning Algorithms](#6)
# 7. [Results](#7)
# 
# 
# 
# 

# <a id="1"> </a>
# ## INTRODUCTION
# 
# * Chronic kidney disease (CKD) is an important public health problem worldwide, especially for underdeveloped countries. Chronic kidney disease means that the kidney is not working as expected and cannot filter blood properly. Approximately 10% of the world's population suffers from this disease and millions die every year. Recently, the number of patients who have reached renal insufficiency is increasing, which necessitates kidney transplant or dialysis. CKD does not show any symptoms in its early stages. The only way to find out if the patient has kidney disease is by testing. Early detection of CKD in its early stages can help the patient receive effective treatment. 
# * The aim of this study is to analyze the methods and compare their accuracy values by using 6 different machine learning methods.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from imblearn.metrics import sensitivity_score
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from sklearn.model_selection import GridSearchCV , KFold , cross_val_score, ShuffleSplit, cross_validate

from collections import Counter
plt.style.use("seaborn-muted")
# Input data files are available in the read-only "../../../input/mansoordaku_ckdisease/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import warnings
# ignore warnings
warnings.filterwarnings("ignore")
from subprocess import check_output
print(check_output(["ls", "../../../input/mansoordaku_ckdisease"]).decode("utf8"))
import os
for dirname, _, filenames in os.walk("../../../input/mansoordaku_ckdisease"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a id="2"> </a>
# ## LOAD AND CHECK DATA

# In[ ]:


#upload dataset
df = pd.read_csv("../../../input/mansoordaku_ckdisease/kidney_disease.csv") 


# In[ ]:


#info about dataset
df.info() 


# In[ ]:


#first five rows of dataset
df.head(10) 


# In[ ]:


#drop id column
df.drop(["id"],axis=1,inplace=True) 


# In[ ]:


#convert to numeric data type
df.pcv = pd.to_numeric(df.pcv, errors='coerce')
df.wc = pd.to_numeric(df.wc, errors='coerce')
df.rc = pd.to_numeric(df.rc, errors='coerce')


# In[ ]:


#statistical information of the features used in the data set
df.describe()


# In[ ]:


#correlation between the features used in the data set
df.corr()


# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(12, 12))
print()
print()


# <a id="3"> </a>
# ## OUTLIER DETECTION

# In[ ]:


#detect outliers
def detect_outliers(df,features):
    outlier_indices=[]
    
    for c in features:
        Q1=np.percentile(df[c],25) #1st quartile
        Q3=np.percentile(df[c],75) #3rd quartile
        IQR=Q3-Q1                  #IQR
        outlier_step=IQR*1.5       #Outlier step
        outlier_list_col=df[(df[c]<Q1-outlier_step) | (df[c]>Q3 + outlier_step)].index #Detect outlier and their indeces
        outlier_indices.extend(outlier_list_col) #Store indeces
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers=list(i for i,v in outlier_indices.items() if v>2)
    
    return multiple_outliers


# In[ ]:


#check if I have outliers
df.loc[detect_outliers(df,["age","bp","sg","al","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc"])]


# <a id="4"> </a>
# ## FILL MISSING VALUE

# In[ ]:


#number of missing values in features
df.isnull().sum()


# In[ ]:


#another way to show missing data

print()
#plt.grid()
#plt.title("Number of Missing Values")


# In[ ]:


#show missing data
import missingno as msno

msno.matrix(df)
print()


# In[ ]:


#show missing data
msno.bar(df)
print()


# In[ ]:


#how missing data in age 
df[df["age"].isnull()]


# In[ ]:


#fill missing data with mean value
df["bgr"]= df["bgr"].fillna(np.mean(df["bgr"]))
df["bu"]= df["bu"].fillna(np.mean(df["bu"]))
df["sc"]= df["sc"].fillna(np.mean(df["sc"]))
df["sod"]= df["sod"].fillna(np.mean(df["sod"]))
df["pot"]= df["pot"].fillna(np.mean(df["pot"]))
df["hemo"]= df["hemo"].fillna(np.mean(df["hemo"]))
df["pcv"]= df["pcv"].fillna(np.mean(df["pcv"]))
df["wc"]= df["wc"].fillna(np.mean(df["wc"]))
df["rc"]= df["rc"].fillna(np.mean(df["rc"]))


# In[ ]:


#The number "1" is indicated by "ckd" (the condition of kidney disease) and the number 
#"0" is indicated by "notckd" (the state of the absence of kidney disease).
df["classification"] = [1 if i == "ckd" else 0 for i in df["classification"]]


# <a id="5"> </a>
# ## DATA VISUALIZATION

# In[ ]:


sns.countplot(df.classification)
plt.xlabel('Chronic Kidney Disease')
plt.title("Classification",fontsize=15)
print()


# In[ ]:


print()
#print()


# In[ ]:


#blood pressure-frequency graph
sns.factorplot(data=df, x='bp', kind= 'count',size=6,aspect=2)


# In[ ]:


#density-frequency graph
sns.factorplot(data=df, x='sg', kind= 'count',size=6,aspect=2)


# In[ ]:


#albumin-frequency graph
sns.factorplot(data=df, x='al', kind= 'count',size=6,aspect=2)


# In[ ]:


#sugar-frequency graph
sns.factorplot(data=df, x='su', kind= 'count',size=6,aspect=2)


# In[ ]:


df['dm'] = df['dm'].replace(to_replace={'\tno':'no','\tyes':'yes',' yes':'yes'})
df['cad'] = df['cad'].replace(to_replace='\tno',value='no')


# In[ ]:


#Check the bar graph of categorical data using factorplot
sns.factorplot(data=df, x='rbc', kind= 'count',size=4,aspect=2)
sns.factorplot(data=df, x='pc', kind= 'count',size=4,aspect=2)
sns.factorplot(data=df, x='pcc', kind= 'count',size=4,aspect=2)
sns.factorplot(data=df, x='ba', kind= 'count',size=4,aspect=2)

sns.factorplot(data=df, x='pcv', kind= 'count',size=6,aspect=2)
sns.factorplot(data=df, x='wc', kind= 'count',size=10,aspect=2)
sns.factorplot(data=df, x='rc', kind= 'count',size=6,aspect=2)

sns.factorplot(data=df, x='htn', kind= 'count',size=4,aspect=2)
sns.factorplot(data=df, x='dm', kind= 'count',size=4,aspect=2)
sns.factorplot(data=df, x='cad', kind= 'count',size=4,aspect=2)
sns.factorplot(data=df, x='appet', kind= 'count',size=4,aspect=2)
sns.factorplot(data=df, x='pe', kind= 'count',size=4,aspect=2)
sns.factorplot(data=df, x='ane', kind= 'count',size=4,aspect=2)


# In[ ]:


def hist_plot(variable):
    plt.figure(figsize=(9,3))
    plt.hist(df[variable],bins=50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("Age Distribution with Histogram")
    print()


# In[ ]:


numericVar = ["age"]
for n in numericVar:
    hist_plot(n)


# In[ ]:


plt.figure(figsize=(70,25))
plt.legend(loc='upper left')
g = sns.countplot(data = df, x = 'age', hue = 'classification')
g.legend(title = 'Kidney Disease', loc='upper left', bbox_to_anchor=(0.1, 0.5), ncol=1)
g.tick_params(labelsize=20)
plt.setp(g.get_legend().get_texts(), fontsize='32')
plt.setp(g.get_legend().get_title(), fontsize='42')
g.axes.set_title('Graph of the number of patients with chronic kidney disease by age',fontsize=50)
g.set_xlabel('Age',fontsize=40)
g.set_ylabel("Count",fontsize=40)


# In[ ]:


g = sns.FacetGrid(df,col="classification")
g.map(sns.distplot,"age", bins=25)
print()


# In[ ]:


sns.factorplot(x="classification",y="age",data=df,kind="box")
print()


# In[ ]:


age_corr = ['age', 'classification']
age_corr1 = df[age_corr]
age_corr_y = age_corr1[age_corr1['classification'] == 1].groupby(['age']).size().reset_index(name = 'count')
age_corr_y.corr()


# In[ ]:


sns.regplot(data = age_corr_y, x = 'age', y = 'count').set_title("Correlation graph for Age vs chronic kidney disease patient")


# In[ ]:


age_corr_n = age_corr1[age_corr1['classification'] == 0].groupby(['age']).size().reset_index(name = 'count')
age_corr_n.corr()


# In[ ]:


sns.regplot(data = age_corr_n, x = 'age', y = 'count').set_title("Correlation graph for Age vs healthy patient")


# In[ ]:


df2 = df.loc[:,["bp","bgr","sod","pot","pcv"]]
df2.plot()


# In[ ]:


df2.plot(subplots = True)
print()


# In[ ]:


g = sns.jointplot("age", "classification", data=df, size=7,ratio=3, color="r")


# In[ ]:


g = sns.jointplot(df.age, df.classification, kind="kde", size=7)
#pearsonr shows the correlation between two features, 1 if positive , -1 if negative, 0 if no correlation.


# In[ ]:


pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=df2, palette=pal, inner="points")
print()


# In[ ]:


sns.boxplot(x="sg", y="age", hue="classification",data=df, palette="PRGn")
print()


# In[ ]:


g = sns.FacetGrid(df,col="classification",row="sg")
g.map(plt.hist,"age", bins=25)
g.add_legend()
print()


# In[ ]:


#I assigned the value 0 and 1 to the nominal features
df['rbc'] = df.rbc.replace(['normal','abnormal'], ['1', '0'])
df['pc'] = df.pc.replace(['normal','abnormal'], ['1', '0'])
df['pcc'] = df.pcc.replace(['present','notpresent'], ['1', '0'])
df['ba'] = df.ba.replace(['present','notpresent'], ['1', '0'])
df['htn'] = df.htn.replace(['yes','no'], ['1', '0'])
df['dm'] = df.dm.replace(['yes','no'], ['1', '0'])
df['cad'] = df.cad.replace(['yes','no'], ['1', '0'])
df['appet'] = df.appet.replace(['good','poor'], ['1', '0'])
df['pe'] = df.pe.replace(['yes','no'], ['1', '0'])
df['ane'] = df.ane.replace(['yes','no'], ['1', '0'])
df.head()


# In[ ]:


#then I converted them to numeric data type
df.rbc = pd.to_numeric(df.rbc, errors='coerce')
df.pc = pd.to_numeric(df.pc, errors='coerce')
df.pcc = pd.to_numeric(df.pcc, errors='coerce')
df.ba = pd.to_numeric(df.ba, errors='coerce')
df.htn = pd.to_numeric(df.htn, errors='coerce')
df.dm = pd.to_numeric(df.dm, errors='coerce')
df.cad = pd.to_numeric(df.cad, errors='coerce')
df.appet = pd.to_numeric(df.appet, errors='coerce')
df.pe = pd.to_numeric(df.pe, errors='coerce')
df.ane = pd.to_numeric(df.ane, errors='coerce')


# In[ ]:


df.info()


# In[ ]:


#I used the knnimputer method for the remaining missing values
#because some features have specific values that's why I didn't get the mean value.
imputer = KNNImputer(n_neighbors=2)
df_filled = imputer.fit_transform(df)


# In[ ]:


df_filled.tolist()


# In[ ]:


#When we use the knnimputer method, we obtained an array
#so I turned it back into a dataframe.
df2 = pd.DataFrame(data = df_filled)


# In[ ]:


#now I have filled all the features
df2.info()


# In[ ]:


df2.head()


# <a id="6"> </a>
# ## MACHINE LEARNING ALGORITHMS

# In[ ]:


#these variables will be used to show the algorithm name and its successes.
score=[] 
algorithms=[] 
precision=[]
sensitivity=[]
recall=[]
f1score=[]


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

y=df2[24].values
x_data=df2.drop([24],axis=1)
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.3)

dt = DecisionTreeClassifier(criterion="entropy", max_depth = 5, random_state=1)
dt.fit(x_train,y_train)


#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=dt.predict(x_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
print()
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Decision Tree Confusion Matrix")
print()

print("Decision Tree accuracy =",dt.score(x_test,y_test)*100)
score.append(dt.score(x_test,y_test)*100)
algorithms.append("Decision Tree")

print("Decision Tree precision =",precision_score(y_true, y_pred,average = 'macro')*100)
precision.append(precision_score(y_true, y_pred,average = 'macro')*100)

print("Decision Tree sensitivity =",sensitivity_score(y_true, y_pred,average = 'macro')*100)
sensitivity.append(sensitivity_score(y_true, y_pred,average = 'macro')*100)

print("Decision Tree recall =",recall_score(y_true, y_pred,average = 'macro')*100)
recall.append(recall_score(y_true, y_pred,average = 'macro')*100)

print("Decision Tree f1 score =",f1_score(y_true, y_pred,average = 'binary')*100)
f1score.append(f1_score(y_true, y_pred,average = 'binary')*100)



#y_pred_prob = dt.predict_proba(x_test)[:,1]
#fpr, tpr, thresholds = roc_curve(y_true, y_pred)
#roc_auc = auc(fpr, tpr)
## Plot ROC curve
#plt.figure()
#plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
#plt.plot(fpr, tpr)
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic')
#plt.legend(loc="lower right")
#print()


# In[ ]:


from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydot
import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)/Graphviz2.38/bin/'

features = list(df.columns[1:])
features

dot_data = StringIO()
export_graphviz(dt, out_file = dot_data,feature_names = features,filled = True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(x_train,y_train)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=rf.predict(x_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
print()
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Random Forest Confusion Matrix")
print()
print("Random Forest accuracy =",rf.score(x_test,y_test)*100)
score.append(rf.score(x_test,y_test)*100)
algorithms.append("Random Forest")

print("Random Forest precision =",precision_score(y_true, y_pred,average = 'macro')*100)
precision.append(precision_score(y_true, y_pred,average = 'macro')*100)

print("Random Forest sensitivity =",sensitivity_score(y_true, y_pred,average = 'macro')*100)
sensitivity.append(sensitivity_score(y_true, y_pred,average = 'macro')*100)

print("Random Forest recall =",recall_score(y_true, y_pred,average = 'macro')*100)
recall.append(recall_score(y_true, y_pred,average = 'macro')*100)

print("Random Forest f1 score =",f1_score(y_true, y_pred,average = 'binary')*100)
f1score.append(f1_score(y_true, y_pred,average = 'binary')*100)


# In[ ]:


#KNN
from sklearn.neighbors import KNeighborsClassifier

y=df2[24].values
x_data=df2.drop([24],axis=1)

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.3)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
knn.predict(x_test)


#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=knn.predict(x_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
print()
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title(" KNN Confusion Matrix")
print()

print("KNN accuracy =",knn.score(x_test,y_test)*100)
score.append(knn.score(x_test,y_test)*100)
algorithms.append("KNN")

print("KNN precision =",precision_score(y_true, y_pred,average = 'macro')*100)
precision.append(precision_score(y_true, y_pred,average = 'macro')*100)

print("KNN sensitivity =",sensitivity_score(y_true, y_pred,average = 'macro')*100)
sensitivity.append(sensitivity_score(y_true, y_pred,average = 'macro')*100)

print("KNN recall =",recall_score(y_true, y_pred,average = 'macro')*100)
recall.append(recall_score(y_true, y_pred,average = 'macro')*100)

print("KNN f1 score =",f1_score(y_true, y_pred,average = 'binary')*100)
f1score.append(f1_score(y_true, y_pred,average = 'binary')*100)


# In[ ]:


from sklearn.svm import SVC

svm = SVC(random_state=1)
svm.fit(x_train,y_train)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=svm.predict(x_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
print()
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Support Vector Machine Confusion Matrix")
print()

print("SVM accuracy =",svm.score(x_test,y_test)*100)
score.append(svm.score(x_test,y_test)*100)
algorithms.append("Support Vector Machine")

print("SVM precision =",precision_score(y_true, y_pred,average = 'macro')*100)
precision.append(precision_score(y_true, y_pred,average = 'macro')*100)

print("SVM sensitivity =",sensitivity_score(y_true, y_pred,average = 'macro')*100)
sensitivity.append(sensitivity_score(y_true, y_pred,average = 'macro')*100)

print("SVM recall =",recall_score(y_true, y_pred,average = 'macro')*100)
recall.append(recall_score(y_true, y_pred,average = 'macro')*100)

print("SVM f1 score =",f1_score(y_true, y_pred,average = 'binary')*100)
f1score.append(f1_score(y_true, y_pred,average = 'binary')*100)


# In[ ]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train,y_train)

#Confusion Matrix 
from sklearn.metrics import confusion_matrix
y_pred=nb.predict(x_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)

#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
print()
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Naive Bayes Confusion Matrix")
print()

print("Naive Bayes accuracy =",nb.score(x_test,y_test)*100)
score.append(nb.score(x_test,y_test)*100)
algorithms.append("Naive Bayes")

print("Naive Bayes precision =",precision_score(y_true, y_pred,average = 'macro')*100)
precision.append(precision_score(y_true, y_pred,average = 'macro')*100)

print("Naive Bayes sensitivity =",sensitivity_score(y_true, y_pred,average = 'macro')*100)
sensitivity.append(sensitivity_score(y_true, y_pred,average = 'macro')*100)

print("Naive Bayes recall =",recall_score(y_true, y_pred,average = 'macro')*100)
recall.append(recall_score(y_true, y_pred,average = 'macro')*100)

print("Naive Bayes f1 score =",f1_score(y_true, y_pred,average = 'binary')*100)
f1score.append(f1_score(y_true, y_pred,average = 'binary')*100)


# In[ ]:


from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lr.fit(x_train,y_train)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred=lr.predict(x_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)
#Confusion Matrix on Heatmap
f,ax=plt.subplots(figsize=(5,5))
print()
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Logistic Regression Confusion Matrix")
print()

print("Logistic Regression accuracy =",lr.score(x_test,y_test)*100)
score.append(lr.score(x_test,y_test)*100)
algorithms.append("Logistic Regression")

print("Logistic Regression precision =",precision_score(y_true, y_pred,average = 'macro')*100)
precision.append(precision_score(y_true, y_pred,average = 'macro')*100)

print("Logistic Regression sensitivity =",sensitivity_score(y_true, y_pred,average = 'macro')*100)
sensitivity.append(sensitivity_score(y_true, y_pred,average = 'macro')*100)

print("Logistic Regression recall =",recall_score(y_true, y_pred,average = 'macro')*100)
recall.append(recall_score(y_true, y_pred,average = 'macro')*100)

print("Logistic Regression f1 score =",f1_score(y_true, y_pred,average = 'binary')*100)
f1score.append(f1_score(y_true, y_pred,average = 'binary')*100)


# In[ ]:


tuned_parameters = [{'n_estimators':[7,8,9,10,11,12,13,14,15,16],'max_depth':[2,3,4,5,6,None], 'class_weight':[None,{0: 0.33,1:0.67},'balanced'],'random_state':[42]}] 
clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=10,scoring='f1')
clf.fit(x_train, y_train)


print("Detailed classification report:")
y_true, lr_pred = y_test, clf.predict(x_test)
print(classification_report(y_true, lr_pred))

confusion = confusion_matrix(y_test, lr_pred)
print('Confusion Matrix:')
print(confusion)

# Determine the false positive and true positive rates
#fpr,tpr,roc_auc = auc_scorer(clf, x_test, y_test, 'RF')

print('Best parameters:')
print(clf.best_params_)
clf_best = clf.best_estimator_


# In[ ]:


plt.figure(figsize=(12,3))
features = x_test.columns.values.tolist()
importance = clf_best.feature_importances_.tolist()
feature_series = pd.Series(data=importance,index=features)
feature_series.plot.bar()
plt.title('Feature Importance')


# In[ ]:


trace1 = { 'x': algorithms, 'y': score, 'name': 'score', 'type': 'bar'  } 

data = [trace1];
layout = { 'xaxis': {'title': 'Classification Algorithms'}, 'title': 'Comparison of Accuracy of Classification Algorithms' }; 
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[ ]:


trace2 = { 'x': algorithms, 'y': precision, 'name': 'score', 'type': 'bar'  } 

data = [trace2];
layout = { 'xaxis': {'title': 'Classification Algorithms'}, 'title': 'Comparison of Precision of Classification Algorithms' }; 
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[ ]:


trace3 = { 'x': algorithms, 'y': sensitivity, 'name': 'score', 'type': 'bar'  } 

data = [trace3];
layout = { 'xaxis': {'title': 'Classification Algorithms'}, 'title': 'Comparison of Sensitivity of Classification Algorithms' }; 
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[ ]:


trace4 = { 'x': algorithms, 'y': recall, 'name': 'score', 'type': 'bar'  } 

data = [trace4];
layout = { 'xaxis': {'title': 'Classification Algorithms'}, 'title': 'Comparison of Recall of Classification Algorithms' }; 
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[ ]:


trace5 = { 'x': algorithms, 'y': f1score, 'name': 'score', 'type': 'bar'  } 

data = [trace5];
layout = { 'xaxis': {'title': 'Classification Algorithms'}, 'title': 'Comparison of F1 Scores of Classification Algorithms' }; 
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# <a id="7"> </a>
# ## RESULTS 
# 
# * In this study, 24 data recording information of 400 people such as age, blood pressure, density, diabetes were used as attributes. Clinical records were examined to determine whether chronic kidney disease was present or not, and provided a high accuracy rate with machine learning methods.
# 
# * Chronic kidney disease is a disease that hinders the normal functions of the kidney and damages the kidneys. It is one of the common diseases in the world and the prediction of the disease is one of the basic issues in medical diagnosis. Chronic kidney disease is one of the leading causes of death worldwide. Early detection of this disease is very important in terms of health and treatment costs. Many machine learning algorithms have been used in the literature to predict the disease.
# 
# * In the study, six different classifiers were utilized in determining the targeted chronic kidney disease and the best performing classifier was tried to be found. These algorithms were compared on the basis of accuracy, sensitivity, sensitivity, recall and f1 score. When the results were evaluated with the data used in this study, it was seen that the random forest method (with an accuracy of 99.16%) performed better than other classification algorithms.
# 
# * Machine learning tools can be used for timely and accurate diagnosis of chronic kidney disease, helping doctors confirm their diagnostic findings in a relatively short time, thereby helping a doctor to look and diagnose more patients in less time. In future studies, it may be possible to use different algorithms, such as deep learning methods, to predict chronic kidney disease.
