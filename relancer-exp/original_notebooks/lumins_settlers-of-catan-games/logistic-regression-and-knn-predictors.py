#!/usr/bin/env python
# coding: utf-8

# #  Catan Data: Who is going to win?

# In[ ]:


# import pandas and read the file into the system
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
path = "../../../input/lumins_settlers-of-catan-games/catanstats.csv"
catan = pd.read_csv(path)


# ## Create an if statement in order to determine the order of players
# - Only the first player is defined in the original dataset

# In[ ]:


for i in range(0, 199, 4):
    if catan.me[i] == 1:
        catan.me[i + 1] = 2
        catan.me[i + 2] = 3
        catan.me[i + 3] = 4
    elif catan.me[i + 1] == 1:
        catan.me[i] = 4
        catan.me[i + 2] = 2
        catan.me[i + 3] = 3
    elif catan.me[i + 2] == 1:
        catan.me[i] = 3
        catan.me[i + 1] = 4
        catan.me[i + 3] = 2
    elif catan.me[i + 3] == 1:
        catan.me[i] = 2
        catan.me[i + 1] = 3
        catan.me[i + 2] = 4


# ## Rename the settlement positions

# In[ ]:


catan = catan.rename(columns={'settlement1':'set1a', 'Unnamed: 17':'set1b', 'Unnamed: 19':'set1c', 'settlement2':'set2a', 'Unnamed: 23':'set2b', 'Unnamed: 25':'set2c'})


# ## Convert the resources to numerical values

# In[ ]:


# convert 'Unnamed: ##' labels to a numerical variable in new terms (columns)
def resources (new, old):
    catan[new] = catan[old].map({'L':0, 'C':1, 'S':2, 'W':3, 'O':4, 'D': 5, '2L':6, '2C':7, '2S':8, '2W':9, '2O':10, '3G':11, 'B':5})

newre = ['re1a', 're1b', 're1c', 're2a', 're2b', 're2c']
oldre = ['Unnamed: 16', 'Unnamed: 18', 'Unnamed: 20', 'Unnamed: 22', 'Unnamed: 24', 'Unnamed: 26']

for new, old in zip(newre, oldre):
    resources (new, old)
    
# dropping unused columns
catan = catan.drop(['Unnamed: 16', 'Unnamed: 18', 'Unnamed: 20', 'Unnamed: 22', 'Unnamed: 24', 'Unnamed: 26'], axis=1)


# ## Create a term (column) for the total chance of numbers rolled per player per starting settlement

# In[ ]:


# convert the 'settlement' labels into numerical terms (columns)
def numchance (new, old):
    catan[new] = catan[old].map({0:0, 2:1, 3:2, 4:3, 5:4, 6:5, 8:5, 9:4, 10:3, 11:2, 12:1})

newset = ['nc1a', 'nc1b', 'nc1c', 'nc2a', 'nc2b', 'nc2c']
oldset = ['set1a', 'set1b', 'set1c', 'set2a', 'set2b', 'set2c']

for new, old in zip(newset, oldset):
                    numchance(new, old)

catan['totalChance'] = catan[:][['nc1a', 'nc1b', 'nc1c', 'nc2a', 'nc2b', 'nc2c']].sum(axis=1)


# ## Create a 'win/loss'  term (column)

# In[ ]:


## Create a 'win/loss' term (column)
catan['win/loss'] = catan.points.map({0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10: 1, 11: 1, 12: 1})


# ## Create a DataFrame and histogram of the total number of each roll
# - Determine if there were enough rolls to validate the central limit theorem
# - Be sure to **divide by 4** since all the rolls are the same for each game by rolls

# In[ ]:


sums = pd.DataFrame(catan[['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']].sum()/4, columns=['totals'])
sums['rolls'] = range(2, 13)


# In[ ]:


# import matplotlib.pyplot (scientific plotting library)
import matplotlib.pyplot as plt
import numpy as np

print()

bins = np.arange(2, 14) - 0.5
plt.hist(sums['rolls'], weights=sums['totals'], bins=bins)
plt.xticks(range(2, 13))
plt.xlabel('Dice Roll')
plt.ylabel('Frequency')
plt.title('Roll Frequency in 50 Games')
plt.grid(True)


# ## Histogram of the total number of points per each starting position

# In[ ]:


bins = np.arange(1, 6) - 0.5
plt.hist(catan['me'], weights=catan['points'], bins=bins)
plt.xticks(range(1, 5))
plt.ylabel('Points')
plt.xlabel('Player Position')
plt.title('Point Totals for Player Positions')
plt.grid(True)


# ## Histogram of the total roll chances from the 2 starting settlements

# In[ ]:


bins = np.arange(1, 6) - 0.5
plt.hist(catan['me'], weights=catan['totalChance'], bins=bins)
plt.xticks(range(1, 5))
plt.ylabel('Total Roll Chances')
plt.xlabel('Player Position')
plt.title('Roll Chance Totals for Player Positions')
plt.grid(True)


# ## Histogram of total wins by player

# In[ ]:


bins = np.arange(1, 6) - 0.5
plt.hist(catan['me'], weights=catan['win/loss'], bins=bins)
plt.xticks(range(1, 5))
plt.ylabel('Wins')
plt.xlabel('Player Position')
plt.title('Win Totals for Player Positions')
plt.grid(True)


# ## Create and X and y from the dataset

# In[ ]:


X = catan[['me', 'production', 'robberCardsGain', 'totalLoss']]
y = catan['win/loss']


# ## Create testing and training datasets using train_test_split

# In[ ]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)


# ## Create Logistic Regression model
# - Used for **binary outcome** like win or loss

# ### scikit-learn's 4-step modeling pattern: Import, Instantiate, Fit, Predict**

# In[ ]:


# Step 1: Import the model
from sklearn.linear_model import LogisticRegression

# Step 2: Instantiate the model
logreg = LogisticRegression(random_state=5)

# Step 3: Fit the model
logreg.fit(X_train, y_train)

# Step 4: Predict
y_pred = logreg.predict(X_test)


# ## Compute classification accuracy
# - Proportion of correct predictions
# 

# In[ ]:


from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)


# ## Total count of win/loss predictions that were correct

# In[ ]:


(y_pred == y_test).value_counts()


# ## KNeighborsClassifier and K-folds cross-validation

# In[ ]:


# split the dataset using K-folds
from sklearn.cross_validation import KFold, cross_val_score
kf = KFold(len(catan), n_folds=10, shuffle=False)
print(cross_val_score(logreg, X, y, cv=kf).mean())


# In[ ]:


# find the best n_neighbors score
from sklearn.neighbors import KNeighborsClassifier
k_range = range(1, 31)
k_scores = []
for K in k_range:
    knn = KNeighborsClassifier(n_neighbors=K)
    scores = cross_val_score(knn, X, y, cv=100, scoring="accuracy")
    k_scores.append(scores.mean())
print(k_scores)
print(max(k_scores))


# In[ ]:


# confirm results above
knn = KNeighborsClassifier(n_neighbors=18)
cross_val_score(knn, X, y, cv=100, scoring="accuracy").mean()

