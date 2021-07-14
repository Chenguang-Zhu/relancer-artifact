#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, plot_importance
# Input data files are available in the "../../../input/aljarah_xAPI-Edu-Data/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../../../input/aljarah_xAPI-Edu-Data"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df1=pd.read_csv("../../../input/aljarah_xAPI-Edu-Data/xAPI-Edu-Data.csv")
df=pd.read_csv("../../../input/aljarah_xAPI-Edu-Data/xAPI-Edu-Data.csv")


# In[ ]:


pd.crosstab(df['Class'],df['Topic'])


# In[ ]:


sns.countplot(x='Topic',hue='Class',data=df1,palette="muted")


# In[ ]:


df1=pd.read_csv("../../../input/aljarah_xAPI-Edu-Data/xAPI-Edu-Data.csv")
df.head(4)


# In[ ]:


df.columns


# In[ ]:


sns.countplot(x='gender',data=df,hue='NationalITy')


# In[ ]:


sns.countplot(x='gender',data=df,palette="muted")


# In[ ]:


sns.countplot(x="Topic", data=df, palette="muted");


# In[ ]:


df.head(4)


# In[ ]:


print()


# In[ ]:


df.head(3)


# In[ ]:


columns=df.dtypes[df.dtypes=='object'].index
columns


# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
for col in df.columns:
    df[col]=encoder.fit_transform(df[col])
    

df.head(3) 


# In[ ]:


df.head(3)


# In[ ]:


##Co-relation
corr=df.corr()
corr = (corr)
plt.figure(figsize=(14,14))
print()
plt.title('Heatmap of Correlation Matrix')


# In[ ]:


sns.regplot(x='Topic',y='Class',data=df)


# In[ ]:


Y=df['Class']
df=df.drop(['Class'],axis=1)
X=df


# In[ ]:


X.head(3)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 30) 


# In[ ]:


#PCA
#Principle component analysis
from sklearn.decomposition import PCA
pca = PCA()
pa=pca.fit_transform(X)
pa


# In[ ]:


covariance=pca.get_covariance()
explained_variance=pca.explained_variance_
explained_variance


# In[ ]:


with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))
    
    plt.bar(range(16), explained_variance, alpha=0.5, align='center', label='individual explained variance') 
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# In[ ]:


Raised_hand = sns.boxplot(x="Class", y="raisedhands", data=df1)
Raised_hand = sns.swarmplot(x="Class", y="raisedhands", data=df1, color=".15")
print()


# In[ ]:


ax = sns.boxplot(x="Class", y="Discussion", data=df1)
ax = sns.swarmplot(x="Class", y="Discussion", data=df1, color=".25")
print()


# In[ ]:


Anounce_bp = sns.boxplot(x="Class", y="AnnouncementsView", data=df1)
Anounce_bp = sns.swarmplot(x="Class", y="AnnouncementsView", data=df1, color=".25")
print() 


# In[ ]:


X_train.head(3)


# In[ ]:


from sklearn.preprocessing import scale

X_train=scale(X_train)
X_test=scale(X_test)


# In[ ]:


from sklearn.linear_model import Perceptron

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train, Y_train)


# In[ ]:


y_pred = ppn.predict(X_test)
print('Misclassified samples: %d' % (Y_test != y_pred).sum())


# In[ ]:


from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(Y_test, y_pred))


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred))


# In[ ]:


from sklearn.svm import SVC

svm = SVC(kernel='linear', C=2.0, random_state=0)
svm.fit(X_train, Y_train)



# In[ ]:


y_pred_SVM = svm.predict(X_test)
print('Accuracy: %.2f' % accuracy_score(Y_test, y_pred_SVM))
print('Misclassified samples: %d' % (Y_test != y_pred_SVM).sum())


# In[ ]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
#clf = MLPClassifier(solver='lbfgs',alpha=1e-5,random_state=1)
clf = MLPClassifier(solver='lbfgs',alpha=.1,random_state=1)
clf.fit(X_train, Y_train)
scores=cross_val_score(clf,X_test,Y_test,cv=10)


# In[ ]:


clf.score(X_test,Y_test)


# In[ ]:


RF = RandomForestClassifier(n_jobs = -1)
RF.fit(X_train, Y_train)
Y_pred = RF.predict(X_test)
RF.score(X_test,Y_test)
print('Misclassified samples: %d' % (Y_test != Y_pred).sum())


# In[ ]:


xgb = XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=100,seed=10)
xgb.fit(X_train, Y_train)
xgb.predict(X_test)
print('Misclassified samples: %d' % (Y_test != Y_pred).sum())


# In[ ]:


xgb_pred=xgb.predict(X_test)


# In[ ]:


print (classification_report(Y_test,xgb_pred))


# In[ ]:


plot_importance(xgb)

