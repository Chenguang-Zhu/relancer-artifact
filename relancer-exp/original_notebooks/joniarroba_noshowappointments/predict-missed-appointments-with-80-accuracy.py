#!/usr/bin/env python
# coding: utf-8

# I created this workbook for educational basis. Please help to improve it further.I created this workbook for educational purposes so please contribute to improve it further. With time, I will be adding more stuff on bias variance trade-off i.e. validation, cross validation, learning curves etc. so STAY TUNED!! For problem description, please visit [this](http://https://www.kaggle.com/joniarroba/noshowappointments) webpage.
# 

# ### Lets import all the required packages.

# In[ ]:


import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from datetime import datetime
from sklearn import model_selection, cross_validation, ensemble, preprocessing, svm
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import learning_curve, ShuffleSplit, validation_curve
from sklearn.model_selection import GridSearchCV


# ### Importing the data & exploring at high level:

# In[ ]:


app_data = pd.read_csv("../../../input/joniarroba_noshowappointments/KaggleV2-May-2016.csv")


# In[ ]:


app_data.head()


# In[ ]:


app_data.info()


# #### Happy to see no nulls above :)

# In[ ]:


app_data.describe()


# In[ ]:


target = sb.countplot(x="No-show", data=app_data)


# #### Our target variable is very unbalanced. Need to account for the imbalance when training your model.
# 

# ### Encoding No-show variable 

# In[ ]:


app_data['noshow'] = [0 if i == "No" else 1 for i in app_data['No-show']]


# ### Deleting rows with age < 0

# In[ ]:


app_data = app_data[app_data.Age > 0]


# In[ ]:


age_hist = sb.boxplot(x="noshow", y="Age", data=app_data)


# #### There are couple of outliers but those do not seem totally unrealistic to me. Interestingly, mean age of people missed appointment is lower than other group.

# In[ ]:


app_data.groupby('noshow').Age.plot(kind='kde')


# #### Younger people (age < 48 years) are better at showing up on scheduled days but trend reverses after age of 48 years. One of the reasons to categorize this variable.

# In[ ]:


app_data['age_cat'] = pd.qcut(app_data['Age'], 5, labels=False)


# In[ ]:


sb.countplot(x="age_cat", hue = "noshow", data=app_data)


# #### The trend we onserved in the density plot is more visible in above bar plot now.

# ### Getting date from Schedules and Appointment dates

# In[ ]:


app_data['scheduled'] = app_data['ScheduledDay'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))


# In[ ]:


app_data['appointment'] = app_data['AppointmentDay'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))


# In[ ]:


app_data['diff_time'] =  (app_data['appointment'] - app_data['scheduled']).dt.days


# In[ ]:


app_data['hour'] = app_data['scheduled'].dt.hour


# In[ ]:


app_data['weekday'] = app_data['scheduled'].dt.weekday


# In[ ]:


app_data['diff_time'] = [0 if i < 0 else i for i in app_data['diff_time']]


# In[ ]:


sb.boxplot(x="noshow", y="diff_time", data=app_data)


# #### Clearly, people who are missing appoints have larger gap between the scheduled day and appointment day.
# 

# In[ ]:


app_data['diff_time_cat'] = [1 if i < 1 else i for i in app_data['diff_time']]
app_data['diff_time_cat'] = [2 if i > 1 and i <= 7 else i for i in app_data['diff_time_cat']]
app_data['diff_time_cat'] = [3 if i > 7 and i <= 30 else i for i in app_data['diff_time_cat']]
app_data['diff_time_cat'] = [4 if i > 30 else i for i in app_data['diff_time_cat']]


# In[ ]:


sb.countplot(x="diff_time_cat", hue = "noshow", data=app_data)


# #### Higher the time between the appointment, higher the probability of missing appointment.

# In[ ]:


sb.countplot(x="weekday", hue = "noshow", data=app_data)


# In[ ]:


sb.countplot(x="hour", hue = "noshow", data=app_data)


# ### We observed that patients with missed appoitment had more time between the day when the appointment was scheduled and actual appointment time.

# In[ ]:


sb.distplot(app_data.diff_time);


# In[ ]:


sb.countplot(x="Gender", data=app_data)


# In[ ]:


app_data.groupby(['noshow'])['diff_time'].mean()


# In[ ]:


app_data['sex'] = [1 if i == 'M' else 0 for i in app_data['Gender']]


# In[ ]:


app_data['area'] = pd.factorize(app_data['Neighbourhood'])[0]


# In[ ]:


sb.countplot(x="area", data=app_data)


# ### Lets train of model

# In[ ]:


X = app_data[['age_cat','Hipertension', 'Alcoholism', 'Diabetes', 'area', 'Scholarship', 'sex', 'SMS_received', 'hour', 'weekday', 'diff_time_cat']] 


# In[ ]:


corr = X.corr()
print()


# #### We see some correlations here. Diabetes and hypertension was expected. Another with SMS recieved and time difference between the appointment day and scheduled day.

# In[ ]:


y = app_data['noshow']


# In[ ]:


x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=1)


# In[ ]:


###Lets check the final table


# In[ ]:


x_train.head()


# ## Model Selection
# 

# In[ ]:



clf = ensemble.RandomForestClassifier(max_depth = 12, n_estimators= 10, verbose=0, class_weight= {0:1, 1:3})
clf.fit(x_train, y_train)


# In[ ]:


y_pred = clf.predict(x_test)


# In[ ]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


clf.score(x_test, y_test)


# In[ ]:


clf.score(x_train, y_train)


# In[ ]:


roc_auc_score(y_test, y_pred)


# In[ ]:


confusion_matrix(y_test, y_pred)


# In[ ]:


importances = clf.feature_importances_


# In[ ]:


std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0) 
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center") 
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
print()


# ### Lets Standardize the X matrix

# In[ ]:


X_scaled = preprocessing.scale(X)


# In[ ]:


x_train, x_test, y_train, y_test = model_selection.train_test_split(X_scaled, y, test_size=0.30, random_state=50)
clf = ensemble.RandomForestClassifier(max_depth = 4, n_estimators= 5, verbose=1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0) 
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center") 
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
print()


# ## We got very good accuracy to start with. We can improve this model by using learning curves and cross validation. Stay tuned for updated version ..........

# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """ This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`. """ 
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    print()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black") 

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


# Compute confusion matrix
class_names = ['0', '1']
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization') 

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix') 

print()


# # To be continued....

# In[ ]:



