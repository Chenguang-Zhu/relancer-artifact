#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[ ]:


df = pd.read_csv("../../../input/sammy123_lower-back-pain-symptoms-dataset/Dataset_spine.csv",usecols=['Col1','Col2','Col3','Col4','Col5','Col6','Col7','Col8','Col9','Col10','Col11','Col12','Class_att'])


# In[ ]:


df.columns = ['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle', 'sacral_slope','pelvic_radius','degree_spondylolisthesis', 'pelvic_slope','Direct_tilt','thoracic_slope','cervical_tilt', 'sacrum_angle','scoliosis_slope','Class_att'] 


# In[ ]:


df.head()


# In[ ]:


features = df[['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle', 'sacral_slope','pelvic_radius','degree_spondylolisthesis', 'pelvic_slope','Direct_tilt','thoracic_slope','cervical_tilt', 'sacrum_angle','scoliosis_slope']] 


# In[ ]:


targetVars = df.Class_att


# In[ ]:


feature_train,feature_test,target_train,target_test = train_test_split(features, targetVars, test_size=0.3)


# In[ ]:


model = LogisticRegression()


# In[ ]:


fitted_model = model.fit(feature_train, target_train)


# In[ ]:


predictions = fitted_model.predict(feature_test)


# In[ ]:


accuracy_score(target_test,predictions)

