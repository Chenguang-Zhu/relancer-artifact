#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score


# In[ ]:


data=pd.read_csv("../../../input/dragonheir_logistic-regression/Social_Network_Ads.csv")
data.head()


# In[ ]:


data.shape


# In[ ]:


data.info


# In[ ]:


data.columns


# In[ ]:


x=data[[ 'Age', 'EstimatedSalary']]
y=data['Purchased']


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=4)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


logrec=LogisticRegression()
logrec.fit(x_train,y_train)


# In[ ]:


logrec.predict(x_test)
'Accuracy :{:.2f}%'.format(logrec.score(x_test,y_test)*100)

