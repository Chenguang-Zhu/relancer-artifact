#!/usr/bin/env python
# coding: utf-8

# # ⚽ El Clásico 2019/2020
# ### ⬜ Real Madryt vs 🟥 Barcelona?

# <img src="https://cdn.pixabay.com/photo/2014/09/27/10/52/estadio-463349_1280.jpg">

# 💡 **Summary:**  
# [El Clásico](https://en.wikipedia.org/wiki/El_Cl%C3%A1sico) is the name given in ⚽ football to any match between Real Madrid and FC Barcelona. Those two teams are the most dominant ([throught history](https://en.wikipedia.org/wiki/La_Liga#All-time_La_Liga_table)) in the Spanish football league, as well as in the whole Europe ([UEFA Champions League finals](https://en.wikipedia.org/wiki/UEFA_Champions_League#Records_and_statistics)).  
# El Clásico is a highly popular event in Europe. For people in the US, it can be viewed as a European 🏈 [Super Bowl](https://en.wikipedia.org/wiki/Super_Bowl). 😁 
# For the purpose of this project, we will focus only on matches in the Spanish league. There are two El Clásico in a season. In season 2019/2020, the first one will take place on [2019-12-18](https://www.marca.com/en/football/barcelona/2019/10/23/5dae7335ca4741971c8b4600.html) at Barcelona Stadium (Camp Nou). The second will take place on 2020-03-01 at Real Madrid Stadium (Santiago Bernabéu).   
# But who will 🏆 win in each of those matches? Let's use deep learning to predict the winner based on the historical results of each match in the Spanish football league.
# 
# P.S. Please, don't bet any real money based on those predictions. 😄 
# 
# 🔢 **Dataset:**    
# Football matches of 1st and 2nd division from season 1970-71 to 2016-17.  
# The model is trained on matches from division nr 1.
# 
# 🧰🤖 **Tools and techniques:**  
# Regression problem for predicting the score difference in a match. The positive sign indicates that the team_1 scored more goals, negative that the team_2 scored more goals.   
# Keras using Model() implementation. Separate model for Categorical Embedding to create a Team Strenght Model. Next merging Team Strenght Model with other input layers. Following those steps is Dense, Dropout, Dense and Output layer.   
# Creating custom loss function with a bigger penalty on predicting the wrong winner.   
# Each feature has its separate input. It was mainly to better show the flow of information in the model. The side effect is that the code is quite lengthy in definitions, fit() method and predict() method. In general, we can combine many inputs f.eg Input((3,)), to make it more compact.
# 
# 
# 🔧 **Possible improvements:**  
# 1. Adding a 'round' feature to tell, which match in seasons it is. I can imagine that there is a different performance of a team in the early season or late season. Head top 4 teams compete to win the season, and top tail 4 teams compete to stay in the first division.
# 2. Offset rank value by minus one season. Currently, there is a data leakage issue, because the rank is calculated based on results after the end of the specific season. In other words, in the first match, the model already knows what was the rank of each team in this season.
# 3. Deep learning RNN model architecture, to better capture the season element as a time series.

# 👍 I hope this notebook will be helpful to you.   
# 💬 I will appreciate any comments, thoughts, ideas and questions in the comment section.  
# 💡 Constructive criticism will be appreciated.  
# 
# 🎉 Have fun! 😁

# 🌍 Notebook published: 19-10-31  
# 🔧 Last update: 19-10-31   
# 👨‍💻 By Artur Górlicki 

# <a id='table_of_contents'></a>
# # 📋 Table of Contents:
# 
# 0. <a href='#section_s0'>🔢 SETTINGS</a>
# 1. <a href='#section_s1'>📥 IMPORT </a>
# 2. <a href='#section_s2'>🧹 TIDY</a>  
# 3. <a href='#section_s3'>🔧 TRANSFORM</a>  
# 3.1 <a href='#section_s31'>Feature engineering</a>  
# 3.2 <a href='#section_s32'>Feature transformation</a>  
# 4. <a href='#section_s4'>🤖 DEEP LEARNING</a>  
# 4.1 <a href='#section_s41'>Categorical Embedding</a>  
# 4.2 <a href='#section_s42'>Model Architecture</a>  
# 4.3 <a href='#section_s43'>Custom Loss Function</a>  
# 4.4 <a href='#section_s44'>Model Results</a>
# 4. <a href='#section_s5'>⚽🔮 EL CLÀSICO 2019/2020</a>  
# 5.1 <a href='#section_s51'>El Clásico -  2019-12-18</a>  
# 5.2 <a href='#section_s52'>El Clásico -  2020-03-01 </a>  
# 5.3 <a href='#section_s53'>📣 And the winner is? 🎉🎉</a>

# <a href='#table_of_contents'>Back to Table of Contents</a>  
# #  
# <a id='section_s0'></a>
# # 0. 🔢 SETTINGS

# In[ ]:


import pandas as pd
import numpy as np
from numpy import unique

RANDOM_STATE = 2019
import random
np.random.seed(RANDOM_STATE) # Set global random_seed

# DATA PREPROCESSING
from sklearn.preprocessing import StandardScaler

# DEEP LEARNING
import keras
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
import keras.losses

# Early Stopping if model is not improving
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience = 5)

from keras.utils import plot_model

import tensorflow as tf
tf.set_random_seed(RANDOM_STATE)


# In[ ]:


# VISUALIZATION
import seaborn as sns
import matplotlib.pyplot as plt
print()
import matplotlib.pylab as plab
plab.rcParams['figure.dpi'] = 200


# <a href='#table_of_contents'>Back to Table of Contents</a>  
# #  
# <a id='section_s1'></a>
# # 1. 📥 IMPORT

# In[ ]:


# Load data
df = pd.read_csv("../../../input/ricardomoya_football-matches-of-spanish-league/FMEL_Dataset.csv", parse_dates = ['date'])
df.tail(10)


# In[ ]:


df.division.value_counts()


# In[ ]:


df = df[df["division"] == 1]


# In[ ]:


df["score_diff"] = df["localGoals"] - df["visitorGoals"]


# In[ ]:


df.head()


# <a href='#table_of_contents'>Back to Table of Contents</a>  
# #  
# <a id='section_s2'></a>
# # 2. 🧹 TIDY

# ### Points table

# Points table will contain the information on each team's performance season-wise. 
# The scoring system works like this, for each won match team gets 3 points, for a draw 1 point and for a lost match 0 points. At the end of the seasons, sum all the points and the winner is the team with the highest amount of points.  
# Based on this table, I will create a rank feature that will indicate the position in the leaderboard at the end of a specific season. 
# 
# Original source of the code to create this table:  
# [*LaLiga Analysis In Depth*](https://www.kaggle.com/spn007/laliga-analysis-in-depth) Sakti Prasad, Kaggle.com (Very helpful, thank you!)

# In[ ]:


df['local_team_won']=df.apply(lambda row: 1 if row['localGoals']>row['visitorGoals'] else 0,axis=1)
df['visitor_team_won']=df.apply(lambda row: 1 if row['localGoals']<row['visitorGoals'] else 0,axis=1)
df['draw']=df.apply(lambda row: 1 if row['localGoals']==row['visitorGoals'] else 0,axis=1)


# In[ ]:


df.tail()


# In[ ]:


laliga=df.copy()


# In[ ]:


a=laliga.groupby(['season','localTeam'])['local_team_won'].sum().reset_index().rename(columns={'localTeam': 'club','local_team_won': 'won'})
b=laliga.groupby(['season','visitorTeam'])['visitor_team_won'].sum().reset_index().rename(columns={'visitorTeam': 'club','visitor_team_won': 'won'})
c=laliga.groupby(['season','localTeam'])['draw'].sum().reset_index().rename(columns={'localTeam': 'club','draw': 'draw'})
d=laliga.groupby(['season','visitorTeam'])['draw'].sum().reset_index().rename(columns={'visitorTeam': 'club','draw': 'draw'})
e=laliga.groupby(['season','localTeam'])['visitor_team_won'].sum().reset_index().rename(columns={'localTeam': 'club','visitor_team_won': 'lost'})
f=laliga.groupby(['season','visitorTeam'])['local_team_won'].sum().reset_index().rename(columns={'visitorTeam': 'club','local_team_won': 'lost'})

point_table=a.merge(b,on=['season','club']).merge(c,on=['season','club']).merge(d,on=['season','club']).merge(e,on=['season','club']).merge(f,on=['season','club'])


# In[ ]:


point_table.head()


# In[ ]:


point_table= point_table.rename(columns={'won_x':'home_win','won_y':'away_win','lost_x':'home_loss','lost_y':'away_loss'})
point_table['matches_won']=point_table.home_win+point_table.away_win
point_table['matches_lost']=point_table.home_loss+point_table.away_loss
point_table['matches_drawn']=point_table.draw_x+point_table.draw_y
point_table=point_table.drop(['draw_x','draw_y'],axis=1)
point_table['total_matches']=point_table.matches_won+point_table.matches_lost+point_table.matches_drawn
point_table['points']=(point_table.matches_won*3)+(point_table.matches_drawn*1)


# In[ ]:


point_table.tail()


# In[ ]:


g=df.groupby(['season','localTeam'])['localGoals'].sum().reset_index().rename(columns={'localTeam': 'club','localGoals': 'home_goals'})
h=df.groupby(['season','visitorTeam'])['visitorGoals'].sum().reset_index().rename(columns={'visitorTeam': 'club','visitorGoals': 'away_goals'})
i=df.groupby(['season','localTeam'])['visitorGoals'].sum().reset_index().rename(columns={'localTeam': 'club','visitorGoals': 'goals_conceded'})
j=df.groupby(['season','visitorTeam'])['localGoals'].sum().reset_index().rename(columns={'visitorTeam': 'club','localGoals': 'goals_conceded'})

point_table=point_table.merge(g,on=['season','club']).merge(h,on=['season','club']).merge(i,on=['season','club']).merge(j,on=['season','club'])

point_table['goals_scored']=point_table.home_goals+point_table.away_goals
point_table['goals_conceded']=point_table.goals_conceded_x+point_table.goals_conceded_y
point_table['goal_difference']=point_table.goals_scored-point_table.goals_conceded
point_table= point_table.drop(['goals_conceded_x','goals_conceded_y'],axis=1)

point_table= point_table.sort_values(by=['season','points','goal_difference']).reset_index().drop('index',axis=1)
point_table = point_table.sort_values(['season', 'points', 'goal_difference'],ascending=False)


# In[ ]:


# Number of teams in each season
print(point_table.season.value_counts().sort_index()[:5])
print(point_table.season.value_counts().sort_index()[-5:])


# Primera División (1996/1997)-Wikipedia  
# https://pl.wikipedia.org/wiki/Primera_Divisi%C3%B3n_(1996/1997)

# In[ ]:


point_table[point_table.season == "1996-97"]


# The points table is more or less the same as the real table. There is an additional challenge, where two teams have the same amount of points (Deportivo and Betis teams). There are special rules for solving this scenario, but it happens not that often, so let's leave it for now. 

# In[ ]:


# Adding a rank column for each team per season
point_table['rank'] = point_table.                        groupby(['season']).                        cumcount() + 1


# In[ ]:


point_table[point_table.season == "1996-97"].head()


# In[ ]:


# In how many seasons Real Madrid was nr 1?
point_table.query('club == "Real Madrid" & rank == "1"').shape[0]


# In[ ]:


# In how many seasons Barcelona was nr 1?
point_table.query('club == "Barcelona" & rank == "1"').shape[0]


# ### Adding rank feature to the original dataframe

# In[ ]:


df.head()


# In[ ]:


columns_rank = ['season','club','rank']


# In[ ]:


rank_df = point_table[columns_rank]
rank_df.head()


# In[ ]:


df_merge1 = df.merge(rank_df, left_on=['season', 'localTeam'], right_on=['season', 'club'], suffixes=('_local', '_Localrank'), how = "left") 
df_merge1.head()


# In[ ]:


# Check if everything was matched
(df_merge1['localTeam'] != df_merge1['club']).sum()


# In[ ]:


df_merge1 = df_merge1.drop(columns=['club'])
df_merge1 = df_merge1.rename(columns={"rank": "localRank"})

df_merge1.head()


# In[ ]:


df_merge2 = df_merge1.merge(rank_df, left_on=['season', 'visitorTeam'], right_on=['season', 'club'], how = "left") 
df_merge2.head()


# In[ ]:


# Check if everything was matched
(df_merge2['visitorTeam'] != df_merge2['club']).sum()


# In[ ]:


df_merge2 = df_merge2.drop(columns=['club'])
df_merge2 = df_merge2.rename(columns={"rank": "visitorRank"})

df_merge2.head()


# In[ ]:


df_merge2['rank_diff'] = df_merge2["localRank"] - df_merge2["visitorRank"]


# In[ ]:


columns_clean = ['date', 'season', 'localTeam', 'visitorTeam', 'score_diff', 'rank_diff']


# In[ ]:


df_clean = df_merge2[columns_clean]
print(df_clean.shape)
df_clean.head()


# <a href='#table_of_contents'>Back to Table of Contents</a>  
# #  
# <a id='section_s3'></a>
# # 3. 🔧 TRANSFORM

# <a href='#table_of_contents'>Back to Table of Contents</a>  
# #  
# <a id='section_s31'></a>
# ### 3.1. Feature engineering

# Let's create a column with the day of the week. 

# In[ ]:


# Add weekday information
df_clean['weekday'] = df_clean['date'].dt.dayofweek + 1 # Monday start from 1, not 0


# In[ ]:


df_clean['weekday'].value_counts().sort_index()


# In Primera Division most of the matches are on weekend - Saturday and Sunday. 

# In[ ]:


df_clean['yearday'] = df_clean['date'].dt.dayofyear


# In[ ]:


sns.distplot(df_clean['yearday'], bins = 52)  
plt.xlim(0, 366) # Maximum of 366 day is a year
print()


# We have a dip around holidays, when the football season is over.

# In[ ]:


df_clean.head()


# <a href='#table_of_contents'>Back to Table of Contents</a>  
# #  
# <a id='section_s32'></a>
# ### 3.2. Feature transformation

# ### Team ID feature

# Let's create a unique team ID, so we can later use it in the Embedding layer. We need a numerical value not a string (team name).

# In[ ]:


# Create list of unique team names
teams_unique = sorted(list(set(df_clean["localTeam"].values).union(df_clean["visitorTeam"].values)))

# Create dictionary: team_name: unique_index
team_id = {team:index for (index, team) in enumerate(teams_unique)}
print(team_id)
print(" ")
# Create reverse dictionary: unique_index:team_name
id_team = {index:team for (index, team) in enumerate(teams_unique)}
print(id_team)


# In[ ]:


# Insert new column
df_clean.insert(loc=4 , column='localTeamID', value=df_clean['localTeam'].map(team_id))
df_clean.insert(loc=5 , column='visitorTeamID', value=df_clean['visitorTeam'].map(team_id))

df_clean.head()    


# ### Season feature

# In[ ]:


df_clean['season_start'] = df_clean['season'].str.split('-').str[0].astype(int)


# In[ ]:


df_clean.tail()


# ### Home feature

# Let's create a variable that will indicate if the team_1 played on his own stadium.   
# Currently data doesn't take it into account. We have a column name local and visitor but it just for naming the column.

# In[ ]:


# Add random variable between 0 and 1
df_clean["home"] = np.random.randint(2, size = df_clean.shape[0])


# In[ ]:


# More or less should be equal
df_clean["home"].value_counts()


# In[ ]:


# If home == 0 then swap visitor with local for team_1
df_clean["team_1"] = np.where(df_clean['home'] == 0, df_clean['visitorTeamID'], df_clean['localTeamID']) 
# If home == 0 then leave visitor and local unchanged
df_clean["team_2"] = np.where(df_clean['home'] == 0, df_clean['localTeamID'] , df_clean['visitorTeamID']) 

# If home == 0 then score_diff needs to be opposite, because of the swap in teams
df_clean["score_diff2"] = np.where(df_clean['home'] == 0, df_clean['score_diff'] * (-1) , df_clean['score_diff']) 

# If home == 0 then rank_diff needs to be opposite, because of the swap in teams
df_clean["rank_diff2"] = np.where(df_clean['home'] == 0, df_clean['rank_diff'] * (-1) , df_clean['rank_diff']) 


# In[ ]:


df_clean.head()


# In[ ]:


# Renaming column names
df_clean = df_clean[["season_start", 'weekday', 'yearday', "team_1", "team_2", "home", "score_diff2", "rank_diff2"]]
df_clean = df_clean.rename(columns = {"score_diff2": "score_diff", "rank_diff2": "rank_diff"})
df_clean.head()


# ### Scaling numerical features

# In[ ]:


df_clean.dtypes


# In[ ]:


numerical_features = ['season_start', 'weekday', 'yearday']


# In[ ]:


scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(df_clean[numerical_features])
scaled_numerical_df = pd.DataFrame(scaled_numerical, columns = numerical_features)


# In[ ]:


# Merge the non-numerical with the scaled numerical data 
df_model = df_clean.merge(right = scaled_numerical_df, how = 'left', left_index = True, right_index = True, suffixes = ('', '_scaled')) 


# In[ ]:


df_model.tail()


# In[ ]:


df_model.info()


# No missing values. :)

# ### Exploratory plots

# In[ ]:


df_model['score_diff'].plot.hist(grid=True, bins=20, rwidth=0.9)
plt.title('Team 1 - Team 2 score')
plt.xlabel('Difference')
plt.ylabel('Occurance')
plt.grid(axis='y', alpha=0.75)


# Quite decent normal distribution. The variance seems to be smaller than in a standard distribution, so we should expect that the model will classify more observations around zero. 

# In[ ]:


df_model['rank_diff'].plot.hist(grid=True, bins=20, rwidth=0.9)
plt.title('Ranking differences')
plt.xlabel('Difference')
plt.ylabel('Occurance')
plt.grid(axis='y', alpha=0.75)


# It seems that the rank feature is calculated properly. There are way more combinations of teams that are close to each other than for example within a 20 place distance, f.eg 1-st team and 20-th team occurs only twice.

# <a href='#table_of_contents'>Back to Table of Contents</a>  
# #  
# <a id='section_s4'></a>
# # 4. 🤖 DEEP LEARNING

# ### Train test split

# Train on almost all seasons. Validation split will be in the .fit method during training.

# In[ ]:


df_model.query('season_start < 2017')['season_start'].value_counts().shape


# Test on last available season.

# In[ ]:


df_model.query('season_start >= 2017')['season_start'].value_counts().shape


# In[ ]:


train_df = df_model.query('season_start < 2017')
test_df = df_model.query('season_start >= 2017')


# In[ ]:


X_train = train_df.drop('score_diff', axis = 1)
X_test = test_df.drop('score_diff', axis = 1)
y_train = train_df['score_diff']
y_test = test_df['score_diff']


# <a href='#table_of_contents'>Back to Table of Contents</a>  
# #  
# <a id='section_s41'></a>
# ### 4.1. Categorical Embedding

# In[ ]:


# Count the unique number of teams
n_teams = len(team_id)

# Create an embedding layer
team_lookup = Embedding(input_dim = n_teams, output_dim = 1, input_length = 1, name = 'Team-Strength') 


# In[ ]:


# Create an input layer for the team ID
teamid_in = Input(shape=(1,))

# Lookup the input in the team strength embedding layer
strength_lookup = team_lookup(teamid_in)

# Flatten the output
strength_lookup_flat = Flatten()(strength_lookup)

# Combine the operations into a single, re-usable model
team_strength_model = Model(inputs = teamid_in, outputs = strength_lookup_flat, name = 'Team-Strength-Model') 


# <a href='#table_of_contents'>Back to Table of Contents</a>  
# #  
# <a id='section_s42'></a>
# ### 4.2. Model Architecture

# In[ ]:


# Create an Input for each team
team_in_1 = Input(shape = (1,), name = 'Team-1') 
team_in_2 = Input(shape = (1,), name = 'Team-2') 

# Create an input for home vs away
home_in = Input(shape = (1,), name = 'Home') 

rank_in = Input(shape = (1,), name = 'Rank') 

season_in = Input(shape = (1,), name = 'Season') 

weekday_in = Input(shape = (1,), name = 'Weekday') 

yearday_in = Input(shape = (1,), name = 'Yearday') 


# Lookup the team inputs in the team strength model
team_1_strength = team_strength_model(team_in_1)
team_2_strength = team_strength_model(team_in_2)

# Combine the team strengths with the other inputs using a Concatenate layer
concat_layer = Concatenate(name = 'Concatenate')([team_1_strength, team_2_strength, home_in, rank_in, season_in, weekday_in, yearday_in]) 

dense_layer1 = Dense(40, activation = 'relu', name = 'Dense')(concat_layer) 
dropout_layer = Dropout(rate = 0.25, seed = RANDOM_STATE, name = "Dropout")(dense_layer1) 
dense_layer2 = Dense(20, activation = 'relu', name = 'Dense2')(dropout_layer) 
output = Dense(1, activation = 'linear', name = 'Output')(dense_layer2) 

# Make a Model
model = Model(inputs = [team_in_1, team_in_2, home_in, rank_in, season_in, weekday_in, yearday_in], outputs = output, name = 'Model') 


# In[ ]:


model.summary()


# In[ ]:


# Plot the model
plot_model(model, to_file='model.png')

# Display the image
data = plt.imread('model.png')
plt.imshow(data)
print()


# <a href='#table_of_contents'>Back to Table of Contents</a>  
# #  
# <a id='section_s43'></a>
# ### 4.3. Custom Loss Function

# We're going to create a custom loss function with a larger penalty for predicting the wrong direction. This will help our neural network learn to at least predict correct winner in the match.

# In[ ]:


# Create loss function
def sign_penalty(y_true, y_pred):
    penalty = 2. # x times bigger penalty for predicting wrong winner
    loss = tf.where(tf.less(y_true * y_pred, 0),                      penalty * tf.abs(y_true - y_pred),                      tf.abs(y_true - y_pred))

    return tf.reduce_mean(loss, axis = -1)

# Enable use of loss with keras
keras.losses.sign_penalty = sign_penalty  
print(keras.losses.sign_penalty)


# In[ ]:


# Compile the model
model.compile(optimizer = keras.optimizers.SGD(lr = 0.05), loss = sign_penalty) 


# In[ ]:


# Fit the model to the dataset
model.fit(x = [train_df['team_1'], train_df['team_2'], train_df['home'], train_df['rank_diff'], train_df['season_start_scaled'], train_df['weekday_scaled'], train_df['yearday_scaled']], y = train_df['score_diff'], epochs = 200, verbose = False, validation_split = 0.3, batch_size = 1024, callbacks = [early_stopping_monitor]) 


# <a href='#table_of_contents'>Back to Table of Contents</a>  
# #  
# <a id='section_s44'></a>
# ### 4.4. Model Results

# In[ ]:


# Save model history on each epoch
history = model.history.history


# In[ ]:


# Loss value in training process
fig = plt.figure(1)
plt.subplot(211)
plt.title('Train and validation loss')
# plot the Train loss
plt.plot(history['loss'], 'r')
plt.xlabel('Epoch')
plt.ylabel('Train' + ' loss:' + str(round(history['loss'][-1], 2)));
plt.subplot(212)
# Plot the Validation loss
plt.plot(history['val_loss'], 'r')
plt.xlabel('Epoch')
plt.ylabel('Validation' + ' loss:' + str(round(history['val_loss'][-1], 2)));
print()


# Validation starts being shaky after 7-th epoch.

# In[ ]:


# Predict the model on the test data
predictions = model.predict([test_df['team_1'], test_df['team_2'], test_df['home'], test_df['rank_diff'], test_df['season_start_scaled'], test_df['weekday_scaled'], test_df['yearday_scaled']]) 


# In[ ]:


list(zip(list(predictions[:, 0]), test_df['score_diff']))[5:10]


# In[ ]:


# Plot
plt.plot( [-5,5],[-5,5], c = "black") # 45 degree line - perfect fit
plt.scatter(test_df['score_diff'], list(predictions[:, 0]))
plt.title('Scatter plot')
plt.xlabel('Actual')
plt.ylabel('Predicted')
print()


# Black line represents a perfect fit. 
# The predictions oscillate in range -2 and 2, while actual have a broader range. As I mentioned in the EDA part, the original distribution is leptokurtic (thin), with not that many high differences of scores. Therefore, the model is predicting observations in a thinner range.

# In[ ]:


pd.Series(predictions[:, 0]).plot.hist(grid=True, bins=20, rwidth=0.9)
plt.title('Team 1 - Team 2 score')
plt.xlabel('Difference')
plt.ylabel('Occurance')
plt.grid(axis='y', alpha=0.75)


# It seems that the model is more towards team_2, because the difference (team_1 - team_2) is wider on the left side (team_2 wins with higher score difference).

# <a href='#table_of_contents'>Back to Table of Contents</a>  
# #  
# <a id='section_s5'></a>
# # 5. ⚽🔮 EL CLÀSICO 2020

# Let's use the model to predict the winner of 2020 El Clásico. 😁   
# **Real Madrid or Barcelona?**

# The output of a model is a number indicating a score difference. Reasonably, if the value is < |1|, there should be a draw. But just to get a clear winner I will choose the sign of value as an indicator of a winner. 

# <a href='#table_of_contents'>Back to Table of Contents</a>  
# #  
# <a id='section_s51'></a>
# ### 5.1. El Clásico -  2019-12-18

# In[ ]:


team_1 = team_id['Real Madrid']
team_2 = team_id['Barcelona']
home = 0 # Camp Nou
rank_diff = 3 - 1 # From 2018/19 La Liga season
season_start = 2019
weekday = pd.to_datetime('2019-12-18').dayofweek + 1
yearday = pd.to_datetime('2019-12-18').dayofyear


# In[ ]:


scaled = scaler.transform([[season_start, weekday, yearday]])


# In[ ]:


game_1 = pd.DataFrame([[season_start, weekday, yearday, team_1, team_2, home, rank_diff, scaled[0,0], scaled[0,1], scaled[0,2]]], columns = X_train.columns) 
game_1.head()


# In[ ]:


# Predict the model on the test data
game_1_predict = model.predict([game_1['team_1'], game_1['team_2'], game_1['home'], game_1['rank_diff'], game_1['season_start_scaled'], game_1['yearday_scaled'], game_1['weekday_scaled']]) 


# <a href='#table_of_contents'>Back to Table of Contents</a>  
# #  
# <a id='section_s52'></a>
# ### 5.2. El Clásico -  2020-03-01 

# In[ ]:


team_1 = team_id['Real Madrid']
team_2 = team_id['Barcelona']
home = 1 # Santiago Bernabéu
rank_diff = 3 - 1 # From 2018/19 La Liga season
season_start = 2019
weekday = pd.to_datetime('2020-03-01').dayofweek + 1
yearday = pd.to_datetime('2020-03-01').dayofyear


# In[ ]:


scaled = scaler.transform([[season_start, weekday, yearday]])


# In[ ]:


game_2 = pd.DataFrame([[season_start, weekday, yearday, team_1, team_2, home, rank_diff, scaled[0,0], scaled[0,1], scaled[0,2]]], columns = X_train.columns) 
game_2.head()


# In[ ]:


# Predict the model on the test data
game_2_predict = model.predict([game_2['team_1'], game_2['team_2'], game_2['home'], game_2['rank_diff'], game_2['season_start_scaled'], game_2['weekday_scaled'], game_2['yearday_scaled']]) 


# <a href='#table_of_contents'>Back to Table of Contents</a>  
# #  
# <a id='section_s53'></a>
# ### 5.3. 📣 And the winner is? 🎉🎉 

# In[ ]:


score_diff = np.asscalar(game_1_predict)
message = 'The winner of the first match on Camp Nou stadium is {0}! \nScore difference is {1:.2f}'
if score_diff > 0:
    print(message.format('Real Madrid', score_diff))
elif score_diff == 0:
    print('Draw!')
else: 
    print(message.format('Barcelona', score_diff))


# In[ ]:


score_diff = np.asscalar(game_2_predict)
message = 'The winner of the second match on Santiago Bernabéu stadium is {0}! \nScore difference is {1:.2f}'
if score_diff > 0:
    print(message.format('Real Madrid', score_diff))
elif score_diff == 0:
    print('Draw!')
else: 
    print(message.format('Barcelona', score_diff))


# Do results seems reasonable? 😁 

# **When I run the the code the results were like this:**  
# The winner of the first match on **Camp Nou** stadium is **Barcelona**! 
# Score difference is -0.33  
# The winner of the second match on **Santiago Bernabéu** stadium is **Real Madrid**! 
# Score difference is 0.03 
# 
# Barcelona will win on its stadium, Real Madrid on its stadium, so it seems like a fair deal. 🤝😁

# 👍 I hope this notebook was helpful to you.  
# 💬 I will appreciate any comments, thoughts, ideas and questions in the comment section.  
# 💡 Constructive criticism will be appreciated.  
# 
# 🎉 Thanks! 😁

# **📚 Learning resources:**  
# [1] [*Advanced Deep Learning with Keras in Python*](https://www.datacamp.com/courses/advanced-deep-learning-with-keras-in-python) DataCamp

# 🔝  <a href='#table_of_contents'>Back to Table of Contents</a>
