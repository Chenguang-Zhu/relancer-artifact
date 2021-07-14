#!/usr/bin/env python
# coding: utf-8

# **Fake News Classification Using DecisionTreeClassifier**

# *In this notebook I train a decision tree classifier to classify fake and genuine news. I would be using the Fake News detection dataset available on kaggle to train the classifier.*

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle # I need this to pickle python objects and save them on disks, because I need them on another project



# In[ ]:


dataset = pd.read_csv("../../../input/jruvika_fake-news-detection/data.csv")


# In[ ]:


print("Total instances : ", len(dataset))
dataset.head()


# In[ ]:


print("Total NaNs:")
dataset.isna().sum()


# The dataset contains 4009 total instances out of which, 21 instances do not have the 'Body' element(NaN). So we drop all the 21 of them so that our dataset is free of NaNs. We are also dropping the 'URLs' column because we want to fit our classifier only on the heading and Body columns.

# In[ ]:


dataset=  dataset.drop(['URLs'], axis=1)
dataset = dataset.dropna()


# In[ ]:


dataset.head()


# Lets seperate our features and targets

# In[ ]:


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# This is the most CPU intensive step. Here we preprocess the Heading and Body columns to get the Bag of Words model. It removes the punctuations, converts all the characters to lowercase and stem them using PorterStemmer

# In[ ]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
ps = PorterStemmer()
for i in range(len(dataset)):
    X[i][0] = ' '.join([ps.stem(word) for word in re.sub('[^a-zA-Z]', ' ', X[i][0]).lower().split() if not word in stopwords.words('english')])
    X[i][1] = ' '.join([ps.stem(word) for word in re.sub('[^a-zA-Z]', ' ', X[i][1]).lower().split() if not word in stopwords.words('english')])


# Now lets initialize CountVectorizer with max_features=5000, so that we only focus on 5000 most frequent terms.
# After that we fit the CountVectorizer objects to Heading and Body columns of X to get the parse matrices

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
mat_body = cv.fit_transform(X[:,1]).todense()
pickle.dump(cv, open(r"cv_body.pkl", "wb"))


# In[ ]:


cv_head = CountVectorizer(max_features=5000)
mat_head = cv_head.fit_transform(X[:,0]).todense()
pickle.dump(cv_head, open(r"cv_head.pkl", "wb"))


# Lets check out the shapes of our matrices :

# In[ ]:


print("Body matrix :", mat_body.shape, "Heading matrix :", mat_head.shape)


# Perfect!
# Stacking the body and heading matrices together to get our feature matrix

# In[ ]:



X_mat = np.hstack(( mat_head, mat_body))


# Splitting the Dataset into training and testing sets

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_mat,y, test_size=0.2, random_state=0)


# Now lets create our DecisionTreeClassifier and fit it into the training set

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier_dtr = DecisionTreeClassifier(criterion='entropy')
classifier_dtr.fit(X_train, y_train)
y_pred_dtr = classifier_dtr.predict(X_test)


# Let's check out the confusion matrix to see how well our model performed.

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_dtr)


# In[ ]:


print(cm)


# Looks pretty good. It classified 29 instances incorrectly out of 798 instances. Thats 96.36 % accuracy!

# In[ ]:


from sklearn.externals import joblib
joblib.dump(classifier_dtr, "classifier_dtr_fakenews_nourl.pkl")


# I am a noob and this is my first attempt at fake news classification. Suggestions are welcome. Thanks for watching my notebook. If you like it please upvote.
