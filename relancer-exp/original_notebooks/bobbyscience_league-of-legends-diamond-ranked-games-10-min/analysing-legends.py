#!/usr/bin/env python
# coding: utf-8

# Let's first do a Exploratory Data Analysis and then do a prediction.

# # EDA

# In[ ]:


from IPython.display import clear_output
print()
clear_output()


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


df = pd.read_csv("../../../input/bobbyscience_league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")
print(f'Columns: {df.columns}')


# Let's plot the histogram for the gold difference, this shows a correlation between blue side winning and the gold difference beeing positive.

# In[ ]:


sns.displot(data=df, x="blueGoldDiff", hue="blueWins")
print()


# Let's see if there is some correlation into wards placed and the average level.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(12,8))

f1 = sns.stripplot(x="redAvgLevel", y="redWardsPlaced", hue="blueWins", data=df, alpha=0.5, ax=ax[0]) 

f2 = sns.stripplot(x="blueAvgLevel", y="blueWardsPlaced", hue="blueWins", data=df, alpha=0.5, ax=ax[1]) 

f1.set_xticklabels(f1.get_xticklabels(), rotation=45)
f2.set_xticklabels(f2.get_xticklabels(), rotation=45)
print()


# As expected if a team kills more they end up winning

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(12,8))

f1 = sns.stripplot(x="redAvgLevel", y="redKills", hue="blueWins", data=df, alpha=0.5, ax=ax[0]) 

f2 = sns.stripplot(x="blueAvgLevel", y="blueKills", hue="blueWins", data=df, alpha=0.5, ax=ax[1]) 

f1.set_xticklabels(f1.get_xticklabels(), rotation=45)
f2.set_xticklabels(f2.get_xticklabels(), rotation=45)
print()


# In[ ]:


f1 = sns.stripplot(x="blueAvgLevel", y="blueExperienceDiff", hue="blueWins", data=df, alpha=0.5,) 

f1.set_xticklabels(f1.get_xticklabels(), rotation=45)
print()


# Let's see the distribution of heralds and dragons.

# In[ ]:


fig, ax = plt.subplots(2,2, figsize=(14,12))

sns.countplot(data=df, x="redDragons", hue="blueWins", ax=ax[0,0])
sns.countplot(data=df, x="redHeralds", hue="blueWins", ax=ax[0,1])
sns.countplot(data=df, x="blueDragons", hue="blueWins", ax=ax[1,0])
sns.countplot(data=df, x="blueHeralds", hue="blueWins", ax=ax[1,1])
print()


# Probably early on the dragons or heralds doesn't matter that much for winning and the skewed distribution towards no heralds or dragons is mainly because they would be taken after 10 minutes.

# We see an inverse correlation between the Gold diferential for the blue team and the Average level for the red team.

# In[ ]:


ax = sns.catplot(x="redAvgLevel", y="blueGoldDiff", hue="blueWins", data=df, alpha=0.3) 
ax.set_xticklabels(rotation=45)
print()


# # Analysing how a team lost with the gold advantage

# We see that there are games that the blue team has an gold advantage and still loses, can we find why searching on other variables?

# In[ ]:


df_bl = df.query('blueWins != 0 & blueGoldDiff > 0')


# In[ ]:


print(f'Red Average Level: {df_bl.redAvgLevel.mean()}')
print(f'Blue Average Level: {df_bl.blueAvgLevel.mean()}')
print(f'Red Average Kills: {df_bl.redKills.mean()}')
print(f'Blue Average Kills: {df_bl.blueKills.mean()}')


# Even though the blue level and gold are higher than the red team, they still lose the game. Let's check the Herald and Dragons for each team:

# In[ ]:


fig, ax = plt.subplots(2,2, figsize=(14,12))

sns.countplot(data=df_bl, x="redDragons", ax=ax[0,0])
sns.countplot(data=df_bl, x="redHeralds", ax=ax[0,1])
sns.countplot(data=df_bl, x="blueDragons", ax=ax[1,0])
sns.countplot(data=df_bl, x="blueHeralds", ax=ax[1,1])
print()


# It seems that I cant' find a exact reason why they loose the game, probably something that occurs after 10 minutes.

# # Useful stats from the data

# In[ ]:


print(f"Probability of winning when you have a gold lead on the blue side: {np.round(len(df.query('blueWins == 1 & blueGoldDiff > 0'))/len(df),3)}")
print(f"Probability of losing when you have a gold lead on the blue side: {np.round(len(df.query('blueWins != 1 & blueGoldDiff > 0'))/len(df),3)}")
print(f"Probability of winning when you have a gold lead on the red side: {np.round(len(df.query('blueWins != 1 & blueGoldDiff < 0'))/len(df),3)}")
print(f"Probability of losing when you have a gold lead on the red side: {np.round(len(df.query('blueWins == 1 & blueGoldDiff < 0'))/len(df),3)}")


# We see that the winning/losing percentage is approximately equal independent of size. 

# # Prediction
# 
# For the prediction, let's split the features of the blue and red team.

# In[ ]:


red_features = [f for f in df.columns if "red" in f]
blue_features = [f for f in df.columns if "blue" in f]


# Let's predict if the blue team wins or loses based on their features.

# In[ ]:


blue_df = df[blue_features]
blue_df.head()


# Let's split the validation and training data:

# In[ ]:


x = blue_df.drop("blueWins", axis=1)
y = blue_df["blueWins"]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# # Random Forest Classifier
# 
# Firstly, let's use a random forest classifier for predicting wins for the blue team.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=0)
randomforest.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import roc_curve, auc

def roc(y_test, y_pred, model_name, title="ROC"):
    """Creates and plots the roc for a model. """ 
    
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label=f'{model_name} ROC curve area = {roc_auc:0.2f}') 
    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")


# For checking if the model works well we will use the ROC curve, if the Area Under the Curve (AUC) is greater than .50, we know that our model works better than a random choice model.

# In[ ]:


from sklearn.metrics import average_precision_score
y_pred_RF = randomforest.predict_proba(X_test)
print(f"Accuracy: {np.around(sum(np.argmax(y_pred_RF, axis=1) == y_test)/len(y_test)*100,1)}%")
average_precision = average_precision_score(y_test, np.argmax(y_pred_RF, axis=1))
roc(y_test, y_pred_RF[:,1], "Random Forest")


# Thus with a simple Random Forest model we can get a robust accuracy of 72% predicting if the blue team wins.
