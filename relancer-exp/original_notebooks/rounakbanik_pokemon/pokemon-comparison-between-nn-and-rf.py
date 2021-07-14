#!/usr/bin/env python
# coding: utf-8

# # Notebook to compare performance between Neural Network and Random Forest
# 
# High-level summary:
#     1. RF essentially creates many decision-trees, and then picks the value that the majority of the trees produce (i.e. a 'forest' of trees). Weighting is visible.
#     2. Neural Network (Multi-layer Perceptron) has multiple layers, paths and activation functions neurons (i.e. a 'web' or 'network' of 'nodes') - which may in parallel traverse multiple paths. The middle layers are hidden and a 'black box'. It is feed forward (and doesn't go backwards).
#     
# Visually:
# 
# Random Forest:
# Courtesy of: https://medium.com/@williamkoehrsen/random-forest-simple-explanation-377895a60d2d
# 
# ![Random Forest](https://miro.medium.com/max/925/1*i0o8mjFfCn-uD79-F1Cqkw.png)
# 
# Neural Network:
# Courtesy of: https://cs231n.github.io/neural-networks-1/
# 
# ![Neural Network](https://cs231n.github.io/assets/nn1/neural_net2.jpeg)

# # Firstly, let's get the dataset and do some EDA, data integration and wrangling
# Courtesy of: https://www.kaggle.com/rounakbanik/pokemon

# In[ ]:


import pandas as pd
import numpy as np
import pandas_profiling

#Suppress warnings
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# Import data
pokemon = pd.read_csv("../../../input/rounakbanik_pokemon/pokemon.csv")

pokemon.head(10)


# In[ ]:


# Next let's do some EDA
pokemon.profile_report(style={'full_width':True})


# In[ ]:


pokemon.describe().transpose()


# In[ ]:


pokemon.dtypes


# You can see the following characteristics:
# 1. 588 different categories
# 2. 731 pokemon total
# 3. 70 legendary pokemon
# 
# As categorical data, you would not expect any correlation etc.

# In[ ]:


# We will drop abilities, for simplicity purposes
pokemon.drop(columns='abilities', inplace=True)

# Drop missing values
pokemon = pokemon.dropna(axis=1) #Rows with NaN

pokemon


# In[ ]:


# Next we need to split feature vs labels/targets
# We will arrayise the features

# Labels are the values we want to predict - in this case, whether a pokemon is legendary
labels = np.array(pokemon['is_legendary'])

# Remove the labels from the features
features = pokemon.drop('is_legendary', axis = 1) # axis 1 refers to the columns
feature_names = list(pokemon.drop('is_legendary', axis = 1).columns) # Get feature names

# While data is already in sparse matrix for many aspects, one-hot encoding required for remaining string values
pokemon_preprocessed = pd.get_dummies(features)

# Convert to numpy array
features = np.array(pokemon_preprocessed)

pokemon_preprocessed.head()


# In[ ]:


# Create training vs test data
# We'll use a 70/30 split
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features,labels,test_size=0.30 ,random_state=42 )

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# In[ ]:


#NN are sensitive to feature scaling, so we'll scale our data - only train data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(train_features)

# Now apply the transformations to the data:
train_features = scaler.transform(train_features)
test_features = scaler.transform(test_features)


# # First let's do it with NN - we won't use any tweaking - just use data as is

# In[ ]:


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs' ,alpha=1e-5 ,hidden_layer_sizes=(2,100) ,random_state=1 )

# Fit training data to NN model
model_NN = clf.fit(train_features, train_labels)


# In[ ]:


# Predict using test data with NN model 
predict_NN = model_NN.predict(test_features)


# In[ ]:


# Evaluate results - show confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class_names = ['Is Legendary', 'Is not Legendary']
cm = confusion_matrix(predict_NN, test_labels)

# Reconvert back to DF
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

# Create seaborn heatmap plot
fig_NN = plt.figure() 

plt.ylabel('True label')
plt.xlabel('Predicted label')

heatmap = sns.heatmap(df_cm, annot=True, fmt="d",cmap="Blues")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14, color='black')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14, color='black')

fig_NN.show()


# In[ ]:


from sklearn.metrics import accuracy_score

print("Accuracy:")
print(accuracy_score(test_labels, predict_NN))


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(test_labels, predict_NN))


# # Next, let's do it with RF

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100 ,max_depth=5 ,random_state=0 )

# Fit training data to RF model
model_RF = clf.fit(train_features, train_labels)


# In[ ]:


predict_RF = model_RF.predict(test_features)


# In[ ]:


# Unlike NN, in RF we can peek under the hood
# Print out 1st tree in the forest
from sklearn.tree import export_graphviz
from graphviz import Source
from IPython.display import Image

export_graphviz(model_RF.estimators_[0],out_file='1_tree_limited.dot',feature_names=(pokemon_preprocessed.columns) ,class_names = ['Is Legendary', 'Is not Legendary'],filled = True)

Image(filename = '1_tree_limited.png')


# In[ ]:


# Print out 10th tree in the forest
export_graphviz(model_RF.estimators_[9],out_file='10_tree_limited.dot',feature_names=(pokemon_preprocessed.columns) ,class_names = ['Is Legendary', 'Is not Legendary'],filled = True)

Image(filename = '10_tree_limited.png')


# In[ ]:


# Evaluate results - show confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class_names = ['Is Legendary', 'Is not Legendary']
cm = confusion_matrix(predict_RF, test_labels)

# Reconvert back to DF
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

# Create seaborn heatmap plot
fig_RF = plt.figure() 

plt.ylabel('True label')
plt.xlabel('Predicted label')

heatmap = sns.heatmap(df_cm, annot=True, fmt="d",cmap="Blues")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14, color='black')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14, color='black')

fig_RF.show()


# In[ ]:


from sklearn.metrics import accuracy_score

print("Accuracy:")
print(accuracy_score(test_labels, predict_RF))


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(test_labels, predict_RF))


# # Final conclusion
# RF performs better with tabular format in this circumstance and has better explainability (you can see the decision trees).
# 
# The Confusion matrices are as follows:
# 
# 

# In[ ]:


print("NN accuracy: ")
print(round(accuracy_score(predict_NN, test_labels) * 100, 2), '%')


# In[ ]:


fig_NN


# In[ ]:


print("RF accuracy: ")
print(round(accuracy_score(predict_RF, test_labels) * 100, 2) ,'%')


# In[ ]:


fig_RF


# However, the data is very skewered and therefore these results are questionable -
# 70 out of 801 Pokemon are legendary (per EDA profile before)
# 731 out of 801 Pokemon are not legendary (per EDA profile before)
# 
# Therefore, a baseline model of just only guessing 'Is not Legendary' would already achieve a 91.3% accuracy.

# # In conclusion, a baseline model of just blindly guessing No would be better than both ML models...
