#!/usr/bin/env python
# coding: utf-8

# 課題2 Car Fuel Consumption

# In[21]:


print()
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_absolute_error 

#小数点がカンマで表記されているため、decimalパラメータを付与
df = pd.read_csv("../../../input/anderas_car-consume/measurements.csv",decimal=',')
df.head()


# In[22]:


#欠損値の確認
df.info()


# temp_inside, gas_type を説明変数として使用したいが、temp_insideは欠損値があり、gas_typeは文字データになっているので、それぞれ対処する。

# In[23]:


# 欠損値は欠損していないデータの平均とする
tmp_temp_inside = df["temp_inside"].dropna()
tmp_temp_inside = tmp_temp_inside.astype(np.float)
tmp_temp_inside_avg = tmp_temp_inside.mean()
df["temp_inside"] = tmp_temp_inside
df["temp_inside"] = df["temp_inside"].fillna(tmp_temp_inside_avg)


# In[24]:


#変数gas_typeをone hot encoding
def distinctGasType(x):
    if x == "E10":
        return 0
    else:
        return 1

df["gas_type_ohe"] = df["gas_type"].apply(lambda x: distinctGasType(x))


# 目的変数と説明変数の関係を確認するためのグラフ作成

# In[25]:


pd.plotting.scatter_matrix(df, figsize=(10,10))
print()


# In[26]:


print()
print()


# データを学習用とテスト用に分割

# In[27]:


from sklearn.model_selection import train_test_split
X = df.drop(columns=['consume','gas_type','specials','refill liters','refill gas']).values
y = df['consume'].values.reshape(-1,1)
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)


# 標準化を行う

# In[28]:


from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_norm = X_train
X_test_norm = X_test
X_train_norm[:,:4] = stdsc.fit_transform(X_train[:,:4])
X_test_norm[:,:4] = stdsc.transform(X_test[:,:4])


# 線形回帰を用いて学習する

# In[29]:


#パラメータ決定
#test_size = 0.2
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

#線形回帰
regr = LinearRegression(fit_intercept=True)
regr.fit(X_train_norm, y_train)
y_pred = regr.predict(X_test_norm)

df_graph = pd.DataFrame()
df_graph['class'] = ['distance','speed','temp_in','temp_out','AC','rain','sun', 'gas_type_ohe']
df_graph['coef']  = regr.coef_.ravel()
sns.set()
sns.catplot(data=df_graph,x='class',y='coef',kind='bar',size=3, aspect=2.5)


# In[30]:


mae = mean_absolute_error(y_test, y_pred) 
print("MAE = %s"%round(mae,3) )


# 決定木

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print()
import seaborn as sns
import graphviz
#import pydotplus
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeRegressor, export_graphviz


# In[32]:


#X_train = df[["x1","x2"]].values
#y_train = df["label"].values
clf = DecisionTreeRegressor(max_depth=None, min_samples_split=3, min_samples_leaf=3, random_state=1234)
clf = clf.fit(X_train_norm, y_train)
print("score=", clf.score(X_train_norm, y_train))
#print(clf.predict(X_test)) #予測したい場合
y_pred = clf.predict(X_test_norm)
mae = mean_absolute_error(y_test, y_pred) 
print("MAE = %s"%round(mae,3) )


# In[33]:


# 説明変数の重要度を出力する
# scikit-learnで算出される重要度は、ある説明変数による不純度の減少量合計である。
print(clf.feature_importances_)
pd.DataFrame(clf.feature_importances_, index=['distance','speed','temp_in','temp_out','AC','rain','sun', 'gas_type_ohe']).plot.bar(figsize=(7,2))
plt.ylabel("Importance")
plt.xlabel("Features")
print()


# 決定木を回帰問題に適用する場合、予測した結果(y_pred)は、訓練データ(y_train)の部分集合？

# In[48]:


count=0
for y_p in y_pred:
    if y_p in y_train:
        count+=1
print("count = %d"%count)
print("size of y_pred = %d"%len(y_pred))


# どうやら違うみたいです

# ランダムフォレスト

# In[35]:


from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(n_estimators=10, max_depth=2, min_samples_leaf=2, min_samples_split=2, random_state=1234) 
clf.fit(X_train_norm, y_train)
print("score=", clf.score(X_train_norm, y_train))

# 説明変数の重要度を出力する
# scikit-learnで算出される重要度は、ある説明変数による不純度の減少量合計である。
print(clf.feature_importances_)
pd.DataFrame(clf.feature_importances_, index=['distance','speed','temp_in','temp_out','AC','rain','sun', 'gas_type_ohe']).plot.bar(figsize=(7,2))
plt.ylabel("Importance")
plt.xlabel("Features")
print()
mae = mean_absolute_error(y_test, y_pred) 
print("MAE = %s"%round(mae,3) )


# アダブースト

# In[36]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3,min_samples_leaf=2,min_samples_split=2,random_state=1234), n_estimators=10, random_state=1234) 
clf.fit(X_train_norm, y_train)
print("score=", clf.score(X_train_norm, y_train))

# 説明変数の重要度を出力する
# scikit-learnで算出される重要度は、ある説明変数による不純度の減少量合計である。
print(clf.feature_importances_)
pd.DataFrame(clf.feature_importances_, index=['distance','speed','temp_in','temp_out','AC','rain','sun', 'gas_type_ohe']).plot.bar(figsize=(7,2))
plt.ylabel("Importance")
plt.xlabel("Features")
print()
mae = mean_absolute_error(y_test, y_pred) 
print("MAE = %s"%round(mae,3) )


# ニューラルネットワーク

# In[37]:


import tensorflow as tf
from tensorflow import keras 
print(tf.__version__)

# import libraries
import numpy as np
import pandas as pds
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes


# In[38]:


# create regression model
def reg_model(optimizer='rmsprop', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(10, input_dim=8, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    # compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# show the model summary
reg_model().summary()


# In[39]:


print()


# In[40]:


mae = mean_absolute_error(y_test, y_pred) 
print("MAE = %s"%round(mae,3) )

