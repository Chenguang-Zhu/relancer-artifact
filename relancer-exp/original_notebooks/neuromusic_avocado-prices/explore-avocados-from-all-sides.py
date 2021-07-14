#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import os
import pandas as pd
import numpy as np
import pylab as pl
from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
import matplotlib.pyplot as pl
import seaborn as sns
data = pd.read_csv("../../../input/neuromusic_avocado-prices/avocado.csv")


# ![![image.png](attachment:image.png)](https://78.media.tumblr.com/6046408b796cf4ef3f14ea5006a2572e/tumblr_oph4kv6j1G1s9f4joo1_500.gif)

# I love avocado. Many thanks to the сreator of this dataset forr a set of data and ideas for graphs.

# In[ ]:


data['Date'] = pd.to_datetime(data['Date'])
data.head()


# Let's check for missing data. I hope that the information about the avocado is full. Well ...

# In[ ]:


data.isnull().sum()


# Luck is definitely on our side! So we have a great set of data and the mood to experiment. Price is always important. The more expensive the better lol. Certainly not for buyers. Let's look at the price distribution of our favorite avocado!

# In[ ]:


pl.figure(figsize=(12,5))
pl.title("Distribution Price")
ax = sns.distplot(data["AveragePrice"], color = 'r')


# In[ ]:


sns.boxplot(y="type", x="AveragePrice", data=data, palette = 'pink')


# Organic avocados are more expensive. This is obvious, because their cultivation is more expensive and we all love natural products and are willing to pay a higher price for them.
# But it is likely that the price of avocado depends not only on the type. Let's look at the price of avocado from different regions in different years.
# Let's start with organic avocados.

# In[ ]:


mask = data['type']=='organic'
g = sns.factorplot('AveragePrice','region',data=data[mask], hue='year', size=13, aspect=0.8, palette='magma', join=False, ) 


# Oh San Francisco, 2017.....
# In 2017, organic avocados were very expensive :(
# Search in Google gave result on this question. In 2017, there was a shortage of avocados. That explains  the price increase!
# ![![image.png](attachment:image.png)](https://padletuploads.blob.core.windows.net/aws/117895962/sgijg1pc_jIydsWIrkmWRw/16743536565e9e2ad31e15cf838ba636.png)

# So, what about conventional type?

# In[ ]:


mask = data['type']=='conventional'
g = sns.factorplot('AveragePrice','region',data=data[mask], hue='year', size=13, aspect=0.8, palette='magma', join=False, ) 


# For obvious reasons, prices are lower. The situation with the price increase in 2017 also affected this type of avocado.

# Organic avocado type is more expensive. And avocado is generally more expensive with each passing year. Objection! lol. We're explorers. Let's see the correlation between these features. First, let's code the categorical attribute - "type".
# 
# 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
dicts = {}

label.fit(data.type.drop_duplicates()) 
dicts['type'] = list(label.classes_)
data.type = label.transform(data.type) 


# In[ ]:


cols = ['AveragePrice','type','year','Total Volume','Total Bags']
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale = 1.7)
print()


# The price of avocado is influenced by the type. Logically. We also see a strong correlation between the features: "Total Bags" and "Total Volume". Also, if you look at the correlation of all the features, you will notice that strongly correlated Small Bags,Large Bag. It is logical but can create problems if we go to predict the price of avocado. I could be wrong, though. I would be grateful to hear the opinion of more experienced people on this issue, thank you!
# 
# Until I decided that can be try classified avocado on type. Organic or not? Experiment so experiment!

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

x = data.drop(['type','region','Date'], axis = 1)
y = data.type

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0)


# I have a warm feeling about logistic regression. Shall we start with this method?

# In[ ]:


logreg =  LogisticRegression(penalty='l1', tol=0.0001).fit(x_train,y_train)
print("LogisticRegression train data score:{:.3f}". format(logreg.score(x_train,y_train))) 
print("LogisticRegression test data score:{:.3f}". format(logreg.score(x_test,y_test))) 


# Good result.  But the values on the training and test set are about the same, which may indicate that we have understudied the model. Next, I'll try to use cross-validation, but for now let's look at the result of the random forest classifier method :)

# In[ ]:


rf =  RandomForestClassifier(n_estimators = 100, random_state = 0, max_features = 2)
rf.fit(x_train,y_train)
precision_rf,recall_rf,thresholds_rf = precision_recall_curve(y_test, rf.predict_proba(x_test)[:,1])

pl.plot(precision_rf,recall_rf,label = 'RF', color = 'c')
close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
pl.plot(precision_rf[close_default_rf], recall_rf[close_default_rf],'^',c = 'k', markersize = 10, label = "Treshold 0.5 RF", fillstyle = "none", mew = 2) 
pl.xlabel("Precision")
pl.ylabel("Recall")
pl.legend(loc = "best")


# The curve of recall and precision suggests that this classifier works great! Let's look at the average precision  and look at the rock curve.

# In[ ]:


from sklearn.metrics import average_precision_score
p_rf = average_precision_score(y_test,rf.predict_proba(x_test)[:,1])
print("Average precision score Random Forest Classifier: {:.3f}". format(p_rf)) 


# Yes, the classifier works fine.

# In[ ]:


fpr_rf,tpr_rf,thresholds_rf = roc_curve(y_test,rf.predict_proba(x_test)[:,1])
pl.plot(fpr_rf,tpr_rf, label = "Roc curve RF")

pl.xlabel("FPR")
pl.ylabel("TPR")

close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
pl.plot(fpr_rf[close_default_rf], tpr_rf[close_default_rf],'^', markersize = 10, label = "Treshold 0.5 RF", fillstyle = "none", c = 'k', mew = 2) 
pl.legend(loc = 4)


# Now let's add cross-validation to train the logistic regression model. Well, why not try  KNeighborsClassifier too? 

# In[ ]:


kfold = 7 
result = {} 
trn_train, trn_test, trg_train, trg_test = cross_validation.train_test_split(x, y, test_size=0.25) 
knn = KNeighborsClassifier(n_neighbors = 200) 
log_reg = LogisticRegression(penalty='l1', tol=0.001) 
scores = cross_validation.cross_val_score(knn, x, y, cv = kfold)
result['KNeighborsClassifier'] = scores.mean()
scores = cross_validation.cross_val_score(log_reg, x, y, cv = kfold)
result['LogisticRegression'] = scores.mean()
pl.clf()

knn_result = knn.fit(trn_train, trg_train).predict_proba(trn_test)
fpr, tpr, thresholds = roc_curve(trg_test, knn_result[:, 1])
roc_auc  = auc(fpr, tpr)
pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('KNeighborsClassifier',roc_auc))

logreg_result = log_reg.fit(trn_train, trg_train).predict_proba(trn_test)
fpr, tpr, thresholds = roc_curve(trg_test, logreg_result[:, 1])
roc_auc  = auc(fpr, tpr)
pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('LogisticRegression',roc_auc))
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.legend(loc=0, fontsize='small')


# I believed in you, logistic regression!
# The RandomForestClassifier has the highest accuracy. But logistic regression and KNeighborsClassifier will also help to know what type of avocado is in front of us.

# Thank you very much for your time spent reading my work. I hope it was interesting and a little helpful for you.
# 
# ![![image.png](attachment:image.png)](https://avatars.mds.yandex.net/get-pdb/752643/3f425efb-7252-4929-af89-7d9a012539af/orig)

# 

# In[ ]:




