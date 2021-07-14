#!/usr/bin/env python
# coding: utf-8

# # with resampling, we get higher accuracy
# 
# ### Churn prediction model 

# **sections**:
# * Exploring the data and visualization
# * Data preprocessing
# * Splitting Data
# * Resampling Training Data
# * Classifiers and Evaluation
# * Tunning the parameters
# * Conclusion
# 

# ### Exploring the data and visualization

# In this section we are going to load data with pandas and explore data by visulization it with seaborn and matplotlib

# In[ ]:


import pandas as pd
import numpy as np 
import seaborn as sb
from IPython.display import display
import matplotlib.pyplot as plt
data = pd.read_csv("../../../input/barelydedicated_bank-customer-churn-modeling/Churn_Modelling.csv")
overview = data.head(10)
target = data['Exited']
print(overview)


# we are removing this features('RowNumber','CustomerId','Surname') this features do not effect with prediction.

# In[ ]:


X = data.iloc[:,3:13]
target = np.array(target)
print(X.head())


# In[ ]:


target_0 = data[data['Exited'] == 0]['Exited'].count()
target_1 = data[data['Exited']== 1]['Exited'].count()
print(target_0,target_1)


# In[ ]:


labels = [0,1]
plt.bar(labels[0],target_0, width=0.1,color = 'red',edgecolor='yellow')
plt.bar(labels[1],target_1,width=0.1,color = 'black',edgecolor='yellow')
plt.legend()


# **Problem in Data**

# we see that our data is imbalanced, because we have 80% of zeros and we have 20% of ones that may make problems with predication.

# ### visualization

# In[ ]:


data.info()


# In[ ]:


data.describe()


# the Data is clear we do not have null values or another types of noise in dataset.

# In[ ]:


fig,axis = plt.subplots(figsize=(8,6))
print()


# In[ ]:


from itertools import chain
countmale = data[data['Gender']=='Male']['Gender'].count()
countfemale=data[data['Gender']=='Female']['Gender'].count()    
fig,aix = plt.subplots(figsize=(8,6))
#print(countmale)
#print(countfemale)
aix = sb.countplot(hue='Exited',y='Geography',data=data)


# we are computing the pairwise correlation between columns because we are having another types in data and we want numbers columns the plot this value to the heatmap to see the correlation.

# In[ ]:


cal= data[data['IsActiveMember']==1].count()
cal2 = data[data['Exited']==1].count()
ave = (cal2/(cal+cal2))*100
va= '%.1f '  % ave[1]
print(va+'%')


# In[ ]:


age = np.array(data['Age'])
fig,axis = plt.subplots(figsize=(8,6))
axis = sb.distplot(age,kde=False,bins=200)


# In[ ]:


axis = sb.jointplot(x='Age',y='Exited',data = data)


# we are seeing that the clients is more between 30 to 50 we are having more clients in this range.

# In[ ]:


g = sb.FacetGrid(data,hue = 'Exited')
(g.map(plt.hist,'Age',edgecolor="w").add_legend())


# In[ ]:


array1 = np.array(data['IsActiveMember'])
array2 = np.array(data['Exited'])
index = len(array1)
count = 0
for i in range(index):
    if(array1[i]==1 and array2[i]==1):
        count +=1
print(count)


# In previous section we are creating code to count all matched 1 with another 1 in exited column ,that is is activate user is effects on the output 735.  

# now we are going to see the important column and the more powerfull column 'geography'. and i want to visualize this column with **plotly**, because it interactive visualization library.

# In[ ]:


France = float(data[data['Geography']=='France']['Geography'].count())
Spain = float(data[data['Geography']=='Spain']['Geography'].count())
Germany = float(data[data['Geography']=='Germany']['Geography'].count())
print(France+Spain+Germany)


# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)


data = dict(type='choropleth', locations=['ESP','FRA','DEU'], colorscale='YlGnBu', text = ['Spain','France','Germany'], z=[France,Spain,Germany], colorbar={'title':'number in each geography'}) 
layout = dict(title='Counting the numbers of each nationality', geo=dict(showframe=False,projection={'type':'natural earth'})) 
choromap = go.Figure(data=[data],layout=layout)


# In[ ]:


iplot(choromap)


# you can easy interact with earth to see the numbers of clients in each country.

# ### Data Preprocessing

# in this section we are making to features preprocessing to be acceptable to feed it to classifier. and in this section below in create label encoder to geography and gender to be numbers that each number refer to name that can not accepted to classifier, example gender have two types female and male this function **LabelEncoder** make this two types in encode like 1 refer to male and 0 refer to female. if we have more types the number is increased. 

# In[ ]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label = LabelEncoder()
X['Geography'] = label.fit_transform(X['Geography'])
X['Gender'] = label.fit_transform(X['Gender'])
print(X['Gender'].head(7))


# In[ ]:


onehotencoding = OneHotEncoder(categorical_features = [1])
X = onehotencoding.fit_transform(X).toarray()
print(X)


# ### Splitting Data

# In[ ]:


from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(X,target,test_size=0.25,random_state=42)


# ### Resampling Training Data

# we have imbalanced data that mean we have the number of samples for class1 is more than class2, the solution for this problem is resampling data and we are resampling data by oversampling it. **Over_sampling** mean that i will generate new samples and add it to the low class that is low than another class.

# In[ ]:


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTEENN
from collections import Counter
go = RandomOverSampler(random_state=42)
train_x_resample,train_y_resample = go.fit_resample(train_x,train_y)
# before resampling the number of each catogorical
print(Counter(train_y).items())
# After resampling the number of each catogorical
print(Counter(train_y_resample).items())
# now let use under sampling 
go1 = ClusterCentroids(random_state=0)
train_x_resample1,train_y_resample1 = go1.fit_resample(train_x,train_y)
# before resampling the number of each catogorical
print(Counter(train_y).items())
# After resampling the number of each catogorical
print(Counter(train_y_resample1).items())
# now let combine two over and under resample
go2 = SMOTEENN(random_state=0) 
train_x_resample2,train_y_resample2 = go2.fit_resample(train_x,train_y)
# before resampling the number of each catogorical
print(Counter(train_y).items())
# After resampling the number of each catogorical
print(Counter(train_y_resample2).items())


# ### Classifiers and Evaluation

# In[ ]:


from sklearn.metrics import accuracy_score,recall_score,f1_score,cohen_kappa_score,precision_score
from time import *
def choose_best(model, train_x , train_y , test_x , test_y):
    result = {}
    
    #for calculate time of fitting data
    start = time()
    model.fit(train_x,train_y)
    end = time()
    result['train_time'] = end-start
    
    #for prediction
    
    start = time()
    test_y_new = model.predict(test_x)
    train_y_new = model.predict(train_x)
    end = time()
    
    result["prediction_time"] = end - start
    
    result['acc_prediction_train'] = accuracy_score(train_y,train_y_new)
    result['recall_prediction_train'] = recall_score(train_y,train_y_new)
    result['f1_score_test'] = f1_score(test_y,test_y_new)
    result['recall_prediction_test'] = recall_score(test_y,test_y_new)
    result['cohen_kappa_score'] = cohen_kappa_score(test_y,test_y_new)
    result['precision_score'] = precision_score(test_y,test_y_new)
    print('name of model {}'.format(model))
    
    return result
    


# In[ ]:


from sklearn.linear_model import LogisticRegression

classifier_1 = LogisticRegression(random_state = 42,solver='lbfgs')
values1 = choose_best(classifier_1,train_x_resample,train_y_resample,test_x,test_y)
values1n = choose_best(classifier_1,train_x_resample1,train_y_resample1,test_x,test_y)
values1nn = choose_best(classifier_1,train_x_resample2,train_y_resample2,test_x,test_y)


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

classifier_2 = AdaBoostClassifier(random_state=42)
values2 = choose_best(classifier_2,train_x_resample,train_y_resample,test_x,test_y)
values2n = choose_best(classifier_2,train_x_resample1,train_y_resample1,test_x,test_y)
values2nn = choose_best(classifier_2,train_x_resample2,train_y_resample2,test_x,test_y)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

classifier_3 = GradientBoostingClassifier()
values3 = choose_best(classifier_3,train_x_resample,train_y_resample,test_x,test_y)
values3n = choose_best(classifier_3,train_x_resample1,train_y_resample1,test_x,test_y)
values3nn = choose_best(classifier_3,train_x_resample2,train_y_resample2,test_x,test_y)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier_4 = RandomForestClassifier(n_estimators=100) #warning 10 to 100
values4 = choose_best(classifier_4,train_x_resample,train_y_resample,test_x,test_y)
values4n = choose_best(classifier_4,train_x_resample1,train_y_resample1,test_x,test_y)
values4nn = choose_best(classifier_4,train_x_resample2,train_y_resample2,test_x,test_y)


# This Dataframe show the result of each classifiers with resampling using : **over_Sampling** method. 

# In[ ]:


moduels = pd.DataFrame({'name_model':["logistic regression","adaboost","gradient boost","random forest"],                       'accuracy_training':[values1["acc_prediction_train"],values2['acc_prediction_train'],values3['acc_prediction_train'],values4['acc_prediction_train']],                       "recall_testing":[values1["recall_prediction_test"],values2["recall_prediction_test"],values3["recall_prediction_test"],values4["recall_prediction_test"]],                        "f1_score":[values1["f1_score_test"],values2["f1_score_test"],values3["f1_score_test"],values4["f1_score_test"]],                        "precision_test":[values1["precision_score"],values2["precision_score"],values3["precision_score"],values4["precision_score"]],                        "kappa_score":[values1["cohen_kappa_score"],values2["cohen_kappa_score"],values3["cohen_kappa_score"],values4["cohen_kappa_score"]],                        "timing_train":[values1["train_time"],values2["train_time"],values3["train_time"],values4["train_time"]],                       "timing_test":[values1["prediction_time"],values2["prediction_time"],values3["prediction_time"],values4["prediction_time"]]})
moduels.sort_values(by =["f1_score"], ascending = False)


# This Dataframe show the result of each classifiers with resampling using : **Under_Sampling** method. 

# In[ ]:


moduels = pd.DataFrame({'name_model':["logistic regression","adaboost","gradient boost","random forest"],                       'accuracy_training':[values1n["acc_prediction_train"],values2n['acc_prediction_train'],values3n['acc_prediction_train'],values4n['acc_prediction_train']],                       "recall_testing":[values1n["recall_prediction_test"],values2n["recall_prediction_test"],values3n["recall_prediction_test"],values4n["recall_prediction_test"]],                        "f1_score":[values1n["f1_score_test"],values2n["f1_score_test"],values3n["f1_score_test"],values4n["f1_score_test"]],                        "kappa_score":[values1n["cohen_kappa_score"],values2n["cohen_kappa_score"],values3n["cohen_kappa_score"],values4n["cohen_kappa_score"]],                        "precision_test":[values1n["precision_score"],values2n["precision_score"],values3n["precision_score"],values4n["precision_score"]],                        "timing_train":[values1n["train_time"],values2n["train_time"],values3n["train_time"],values4n["train_time"]],                       "timing_test":[values1n["prediction_time"],values2n["prediction_time"],values3n["prediction_time"],values4n["prediction_time"]]})
moduels.sort_values(by =["f1_score"], ascending = False)


# This Dataframe show the result of each classifiers with resampling using : **Combination of twos** method. 

# In[ ]:


moduels = pd.DataFrame({'name_model':["logistic regression","adaboost","gradient boost","random forest"],                       'accuracy_training':[values1nn["acc_prediction_train"],values2nn['acc_prediction_train'],values3nn['acc_prediction_train'],values4nn['acc_prediction_train']],                       "recall_testing":[values1nn["recall_prediction_test"],values2nn["recall_prediction_test"],values3nn["recall_prediction_test"],values4nn["recall_prediction_test"]],                        "f1_score":[values1nn["f1_score_test"],values2nn["f1_score_test"],values3nn["f1_score_test"],values4nn["f1_score_test"]],                        "kappa_score":[values1nn["cohen_kappa_score"],values2nn["cohen_kappa_score"],values3nn["cohen_kappa_score"],values4nn["cohen_kappa_score"]],                        "precision_test":[values1nn["precision_score"],values2nn["precision_score"],values3nn["precision_score"],values4nn["precision_score"]],                        "timing_train":[values1nn["train_time"],values2nn["train_time"],values3nn["train_time"],values4nn["train_time"]],                       "timing_test":[values1nn["prediction_time"],values2nn["prediction_time"],values3nn["prediction_time"],values4nn["prediction_time"]]})
moduels.sort_values(by =["f1_score"], ascending = False)


# based on kappa score and f1 score that is best choose to evalute the best model in imbalanced dataset, we arre choose the gradient boost as the classifier for this problem and we are choose the resampling method number **3** that is combination of under_sampler and oversampler.

# ### Tunning the parameters

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score,make_scorer,classification_report,confusion_matrix,roc_auc_score
parameters = [{'loss':['deviance'],'learning_rate':[0.1,0.2,0.3,0.4],'n_estimators':[50,100], 'max_depth':[3,6,10,15]}] 
scorer = make_scorer(fbeta_score,beta=0.5)
grid_search =  GridSearchCV(estimator = classifier_3, param_grid  = parameters, scoring = scorer ,cv = 5)
grid_fit = grid_search.fit(train_x_resample,train_y_resample)
best_accuracy = grid_fit.best_score_
best_para = grid_fit.best_params_
best_clas = grid_fit.best_estimator_
prdict_y  = best_clas.predict(test_x)
score = fbeta_score(test_y,prdict_y,beta=0.5)
print(best_accuracy,best_para,score)


# In[ ]:


confusionMatrix = confusion_matrix(test_y,prdict_y)
print()


# In[ ]:


print(classification_report(test_y,prdict_y))


# In[ ]:


roc = roc_auc_score(test_y,prdict_y)
print(roc)


# ### Conclusion

# In this problem we faced on more challenge and beat it, first problem is imbalanced dataset and we define three methods of resampling:
# - overSampling: In this method add samples in lower class and make it equal to class2, the disadantage is : we can make overfitting to our model.
# - underSampling: In this method add to lower class more samples repeted samples to lower class to be equaled to another class disadventage:that we are lossing more important information.
# - combination oversampling and undersampling(the best): This method is the best one because it combine the advantage of first method and advantage second method add and remove samples to be nearly equal to each other.
# #### then
# we choose the classifier and feed our resampling data to fitting and training on it, then we choose the higher one based on metrics and we choose metrics specified to deal with imbalanced data like: f1 score, kappa score, precision, recall then we create grid search to selected most powerful parameters to classifier.then create confusion and roc curve to see the final result.
