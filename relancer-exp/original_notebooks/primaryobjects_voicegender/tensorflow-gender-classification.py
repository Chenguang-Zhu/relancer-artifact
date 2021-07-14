#!/usr/bin/env python
# coding: utf-8

# ## Gender classification model using [tf.contrib.learn][1]
# 
#  - In this model we use Deep Neural Network Classifier ([DNNClassifier][2])
#  - This kernel is forked from [Evaluation of gender classification model][3] 
#  - The code change is mainly in script's [classification model][4]
# 
# 
#   [1]: https://www.tensorflow.org/versions/r0.11/tutorials/tflearn/index.html#tf-contrib-learn-quickstart
#   [2]: https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.learn.html#DNNClassifier
#   [3]: https://www.kaggle.com/lewuathe/d/primaryobjects/voicegender/evaluation-of-gender-classification-model
#   [4]: https://www.kaggle.io/svf/383443/7c20eed55e778f52cdf83f02d5b581eb/__results__.html#Create-classification-model

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../../../input/primaryobjects_voicegender/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../../../input/primaryobjects_voicegender"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../../../input/primaryobjects_voicegender/voice.csv")
df.head(2)


# In[ ]:


print("Total number of samples: {}".format(df.shape[0]))
print("Number of male: {}".format(df[df.label == 'male'].shape[0]))
print("Number of female: {}".format(df[df.label == 'female'].shape[0]))


# Data set first should be separated with training data and test data for evaluating trained model.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

# Encode label category
# male -> 1
# female -> 0

gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# ----------
# 
# ## Create classification model
# 
#  - We create a pipeline which has preprocessing model and a DNNClassifier model

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=len(X[0]))]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3, model_dir="tf_model") 

pipe_tf = Pipeline([('std_scl', StandardScaler()), ('dnn', classifier)]) 


# In[ ]:


# Fit model.
pipe_tf.fit(X_train, y_train, dnn__steps=2000)


# In[ ]:


from sklearn.metrics import accuracy_score

# Evaluate accuracy.
score = accuracy_score(y_test, pipe_tf.predict(X_test))
print('Accuracy: {0:f}'.format(score))


# In[ ]:




