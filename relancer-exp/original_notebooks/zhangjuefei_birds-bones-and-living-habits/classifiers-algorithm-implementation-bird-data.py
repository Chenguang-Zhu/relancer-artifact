#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

import seaborn as sns
import matplotlib.pyplot as plt



# In[ ]:




data = pd.read_csv("../../../input/zhangjuefei_birds-bones-and-living-habits/bird.csv")



# In[ ]:




data.head(3)



# In[ ]:


data['type'].unique()


# In[ ]:




data.describe()



# In[ ]:




data.isnull().any()



# In[ ]:


data = data.dropna()


# In[ ]:


data.describe()


# In[ ]:


data.isnull().any()



# In[ ]:


data['type'].unique()


# In[ ]:




#we can assign the  SW,W,T,R,P,SO as 0,1,2,3,4,5 respectively for the types .



# In[ ]:


data.type[data.type == 'SW'] = 0
data.type[data.type == 'W'] = 1
data.type[data.type == 'T'] = 2
data.type[data.type == 'R'] = 3
data.type[data.type == 'P'] = 4
data.type[data.type == 'SO'] = 5


# In[ ]:




data['type'].unique()



# In[ ]:




data = data.drop('id',axis=1)



# In[ ]:




data.head(3)



# In[ ]:


y = data['type']


# In[ ]:




x = data.drop('type',axis=1)



# In[ ]:


plt.figure(figsize=(2,1))
print()
print()


# In[ ]:


plt.figure(figsize=(40,20))
cm = np.corrcoef(x.values.T)
sns.set(font_scale = 1.5)
print()
print()


# In[ ]:



x


# In[ ]:




y = y.astype('int')



# In[ ]:




y.head(10)



# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x = scaler.fit_transform(x)


# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)


# Implementing PCA to find key dependent features

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=9)


# In[ ]:


pca.fit(data)


# In[ ]:


print(pca.explained_variance_ratio_)  


# **KNN implementation**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 1).fit(x_train, y_train) 


# In[ ]:


accuracy = knn.score(x_test, y_test) 
print (accuracy) 


# **Implementing Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(x_train, y_train) 
gnb_predictions = gnb.predict(x_test) 


# In[ ]:


accuracy = gnb.score(x_test, y_test) 
print (accuracy)


# **Implementing Kmeans clustering**

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


kmeansclust = KMeans(n_clusters=6,algorithm = 'elkan',max_iter = 40 , n_jobs = 10)
kmeansclust.fit(x_train,y_train)


# In[ ]:


ykmeans = kmeansclust.predict(x_test)


# In[ ]:




from sklearn.metrics import precision_score

print (precision_score(y_test, ykmeans,average='macro'))



# In[ ]:


from sklearn import metrics
confusion_matrix=metrics.confusion_matrix(y_test,ykmeans)
confusion_matrix


# **Implementing LDA**

# In[ ]:




from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
ldaalg = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)
ldaalg.fit(x_train ,y_train)



# In[ ]:


ldaalg.predict(x_test)


# In[ ]:


ldaalg.score(x_train,y_train)


# In[ ]:


ldaalg.score(x_test,y_test)


# **Implementing Decision tree classifier**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier(random_state=0,criterion='entropy')
dectree.fit(x_train,y_train)


# In[ ]:


dectree.predict(x_test)


# In[ ]:


dectree.score(x_train,y_train)


# In[ ]:


dectree.score(x_test,y_test)


# **Implementing SVC using various parametres**

# In[ ]:


from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(x_train, y_train) 
svm_predictions = svm_model_linear.predict(x_test) 


# In[ ]:


svm_model_linear.score(x_train, y_train)


# In[ ]:


svm_model_linear.score(x_test, y_test)


# SVC by checking kernel as rbf,random_state as 0 , gamma as 0.10 and finally C as 10.0 

# In[ ]:


svm_svc = SVC(kernel = 'rbf',random_state = 0 , gamma = 0.10 ,C=10.0)
svm_svc.fit(x_train, y_train) 


# In[ ]:



svm_svc.score(x_train, y_train) 


# In[ ]:


svm_svc.score(x_test, y_test) 


# SVC by checking kernel as rbf,random_state as 0 , gamma as 0.20 and finally C as 10.0

# In[ ]:


svm_svc1 = SVC(kernel = 'rbf',random_state = 0 , gamma = 0.2 ,C=10.0)
svm_svc1.fit(x_train, y_train) 


# In[ ]:



svm_svc1.score(x_train, y_train) 


# In[ ]:


svm_svc1.score(x_test, y_test) 


# SVC by checking kernel as rbf,random_state as 0 , gamma as 2 and finally C as 10.0 

# In[ ]:


svm_svc2 = SVC(kernel = 'rbf',random_state = 0 , gamma = 2 ,C=10.0)
svm_svc2.fit(x_train, y_train) 


# In[ ]:


svm_svc2.score(x_train, y_train) 


# In[ ]:


svm_svc2.score(x_test, y_test) 


# SVC by checking kernel as rbf,random_state as 0 , gamma as 0.35 and finally C as 10.0 

# In[ ]:


svm_svc3 = SVC(kernel = 'rbf',random_state = 0 , gamma = 0.40 ,C=10.0)
svm_svc3.fit(x_train, y_train) 


# In[ ]:


svm_svc3.score(x_train, y_train) 


# In[ ]:


svm_svc3.score(x_test, y_test) 


# SVC by checking kernel as rbf,random_state as 0 , gamma as 0.35 and finally C as 70 

# In[ ]:


svm_svc3 = SVC(kernel = 'linear',random_state = 0 , gamma = 0.35 ,C=70.0)
svm_svc3.fit(x_train, y_train) 


# In[ ]:


svm_svc3.score(x_train, y_train) 


# In[ ]:


svm_svc3.score(x_test, y_test) 


# **Implementing Random forest classifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

random_clf = RandomForestClassifier(n_estimators=300,random_state=1,criterion='entropy',n_jobs=4)
random_clf.fit(x_train,y_train)


# In[ ]:


random_clf.score(x_train,y_train)


# In[ ]:


random_clf.score(x_test,y_test)


# In[ ]:


#We can use grid search in order to find the parameters efficiently , 
#which can be used for this implementation in order to determine the best results for training set

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
pipe_svc = Pipeline([('clf', RandomForestClassifier(random_state=1))])
estim_range = [1,10,50,100,200,300,400,700,1000]
jobs_range = [1,2,3,5,8,10,16]
crit_range = ['gini','entropy']
param_grid = [{'clf__n_estimators': estim_range,'clf__criterion': crit_range ,'clf__n_jobs': jobs_range}]
gs = GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy')
gs = gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params_)



# In[ ]:


random_clf1 = RandomForestClassifier(n_estimators=100,criterion='gini',n_jobs=1)
random_clf1.fit(x_train,y_train)


# In[ ]:



random_clf1.score(x_train,y_train)


# In[ ]:


random_clf1.score(x_test,y_test)


# 
# **Implementing SGD classifier to validate this dataset**

# In[ ]:


from sklearn.linear_model import SGDClassifier
clfsgd = SGDClassifier(loss="log", penalty="l2", max_iter=400)
clfsgd.fit(x_train,y_train)


# In[ ]:


clfsgd.score(x_train,y_train)


# In[ ]:


clfsgd.predict(x_test)


# In[ ]:


clfsgd.score(x_test,y_test)


# **Implementing Logistic regression for this dataset**

# In[ ]:


from sklearn.linear_model import LogisticRegression
logclf = LogisticRegression(random_state=1,multi_class='multinomial',solver='newton-cg',max_iter=300)
logclf.fit(x_train,y_train)


# In[ ]:


logclf.score(x_train,y_train)


# In[ ]:


logclf.score(x_test,y_test)


# **Attempting Agglomerative clustering model**

# In[ ]:


from sklearn.cluster import AgglomerativeClustering
aggclustering = AgglomerativeClustering(n_clusters=6)
aggclustering.fit(x_train,y_train)


# In[ ]:


aggclusterval = aggclustering.fit_predict(x_test)


# In[ ]:


from sklearn.metrics import precision_score

print (precision_score(y_test, aggclusterval,average='macro'))


# **Implementing XGboost concept to check accuracy**

# In[ ]:


import xgboost as xgb

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)


# In[ ]:


#defining parameters 
param = {'max_depth': 6, 'eta': 0.3,'silent': 1,'objective': 'multi:softprob','num_class': 6}  
num_round = 20


# In[ ]:


xgbstimp = xgb.train(param, dtrain, num_round)


# In[ ]:


preds = xgbstimp.predict(dtest)


# In[ ]:


best_preds = np.asarray([np.argmax(line) for line in preds])


# In[ ]:


from sklearn.metrics import precision_score

print (precision_score(y_test, best_preds, average='macro'))


# **Implementing Keras for this dataset **

# In[ ]:


from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle


# In[ ]:


y1 = y


# In[ ]:


x1 = x


# In[ ]:


encoder = LabelEncoder()
encoder.fit(y1)
y1 = encoder.transform(y1)
y1 = np_utils.to_categorical(y1)


# In[ ]:


from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.20)
print (x1_train.shape, y1_train.shape)
print (x1_test.shape, y1_test.shape)


# In[ ]:


#sample y1 here 
y1


# In[ ]:


#selecting the dense with numbe rof neurons 
input_dim = len(data.columns) - 1

model = Sequential()
model.add(Dense(11, input_dim = input_dim , activation = 'relu'))
model.add(Dense(11, activation = 'relu'))
model.add(Dense(11, activation = 'relu'))
model.add(Dense(11, activation = 'relu'))
model.add(Dense(6, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(x1_train, y1_train, epochs = 90, batch_size = 2)

scores = model.evaluate(x1_test, y1_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# 
# **Conclusion**
# Best Algorithm suitable for this is *SVC , SGD , Randomforest ,Keras* implementationwith above 90 % accuracy
# Will be updating further in future 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




