#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


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

data = pd.read_csv("../../../input/aljarah_xAPI-Edu-Data/xAPI-Edu-Data.csv")


# **Dataset**

# In[ ]:


len(data)


# In[ ]:


data.head(4)


# In[ ]:


data.columns


# Attributes
# 1 Gender - student's gender (nominal: 'Male' or 'Female’)
# 
# 2 Nationality- student's nationality (nominal:’ Kuwait’,’ Lebanon’,’ Egypt’,’ SaudiArabia’,’ USA’,’ Jordan’,’ Venezuela’,’ Iran’,’ Tunis’,’ Morocco’,’ Syria’,’ Palestine’,’ Iraq’,’ Lybia’)
# 
# 3 Place of birth- student's Place of birth (nominal:’ Kuwait’,’ Lebanon’,’ Egypt’,’ SaudiArabia’,’ USA’,’ Jordan’,’ Venezuela’,’ Iran’,’ Tunis’,’ Morocco’,’ Syria’,’ Palestine’,’ Iraq’,’ Lybia’)
# 
# 4 Educational Stages- educational level student belongs (nominal: ‘lowerlevel’,’MiddleSchool’,’HighSchool’)
# 
# 5 Grade Levels- grade student belongs (nominal: ‘G-01’, ‘G-02’, ‘G-03’, ‘G-04’, ‘G-05’, ‘G-06’, ‘G-07’, ‘G-08’, ‘G-09’, ‘G-10’, ‘G-11’, ‘G-12 ‘)
# 
# 6 Section ID- classroom student belongs (nominal:’A’,’B’,’C’)
# 
# 7 Topic- course topic (nominal:’ English’,’ Spanish’, ‘French’,’ Arabic’,’ IT’,’ Math’,’ Chemistry’, ‘Biology’, ‘Science’,’ History’,’ Quran’,’ Geology’)
# 
# 8 Semester- school year semester (nominal:’ First’,’ Second’)
# 
# 9 Parent responsible for student (nominal:’mom’,’father’)
# 
# 10 Raised hand- how many times the student raises his/her hand on classroom (numeric:0-100)
# 
# 11- Visited resources- how many times the student visits a course content(numeric:0-100)
# 
# 12 Viewing announcements-how many times the student checks the new announcements(numeric:0-100)
# 
# 13 Discussion groups- how many times the student participate on discussion groups (numeric:0-100)
# 
# 14 Parent Answering Survey- parent answered the surveys which are provided from school or not (nominal:’Yes’,’No’)
# 
# 15 Parent School Satisfaction- the Degree of parent satisfaction from school(nominal:’Yes’,’No’)
# 
# 16 Student Absence Days-the number of absence days for each student (nominal: above-7, under-7)

# **Exploratory data analysis**

# In[ ]:


data['gender'].value_counts()


# **Gender Percentage In Dataset**

# In[ ]:


print('Percentage',data.gender.value_counts(normalize=True))
data.gender.value_counts(normalize=True).plot(kind='bar')


# In[ ]:


data['PlaceofBirth'].value_counts()


# In[ ]:


nationality = sns.countplot(x = 'PlaceofBirth', data=data, palette='Set3')
nationality.set(xlabel='PlaceofBirth',ylabel='count', label= "Students Birth Place")
plt.setp(nationality.get_xticklabels(), rotation=90)
print()


# In[ ]:


pd.crosstab(data['Class'],data['Topic'])


# In[ ]:


sns.countplot(x='StudentAbsenceDays',data = data, hue='Class',palette='dark')
print()


# In[ ]:


P_Satis = sns.countplot(x="ParentschoolSatisfaction",data=data,linewidth=2,edgecolor=sns.color_palette("dark"))


# **Gender Comparison With Parents Relationship **

# In[ ]:


# gender comparison Relationship with Pare
plot = sns.countplot(x='Class', hue='Relation', data=data, order=['L', 'M', 'H'], palette='Set1')
plot.set(xlabel='Class', ylabel='Count', title='Gender comparison')
print()

#educational level student belongs (nominal: ‘lowerlevel’,’MiddleSchool’,’HighSchool’)


# **Pairplot**

# In[ ]:


print()


# In[ ]:


#Graph Analysis Gender vs Place of Birth


# In[ ]:


import networkx as nx

g= nx.Graph()
g = nx.from_pandas_dataframe(data,source='gender',target='PlaceofBirth')
print (nx.info(g))


plt.figure(figsize=(10,10)) 
nx.draw_networkx(g,with_labels=True,node_size=50, alpha=0.5, node_color="blue")
print()


# **Machine Learning Methods**

# In[ ]:


data.dtypes


# **Label Encoding**

# In[ ]:


Features = data.drop('Class',axis=1)
Target = data['Class']
label = LabelEncoder()
Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
for col in Cat_Colums:
    Features[col] = label.fit_transform(Features[col])
    


# **Test and Train Data Split**

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=0.2, random_state=52)


# **Logistic Regression Model**

# In[ ]:


Logit_Model = LogisticRegression()
Logit_Model.fit(X_train,y_train)


# In[ ]:


Prediction = Logit_Model.predict(X_test)
Score = accuracy_score(y_test,Prediction)
Report = classification_report(y_test,Prediction)



# In[ ]:


print(Prediction)


# In[ ]:


print(Score)


# In[ ]:


print(Report)


# **XGBoost**

# In[ ]:


xgb = XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=100,seed=10)
xgb_pred = xgb.fit(X_train, y_train).predict(X_test)


# In[ ]:


print (classification_report(y_test,xgb_pred))


# In[ ]:


print(accuracy_score(y_test,xgb_pred))


# In[ ]:


plot_importance(xgb)


# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:




# In[ ]:




# In[ ]:





# In[ ]:


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

