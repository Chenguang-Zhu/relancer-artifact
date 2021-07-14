#!/usr/bin/env python
# coding: utf-8

# # Car Fuel Consumption
# 被説明変数：`Consume`  
# 説明変数：`distance`, `speed`, `temp_inside`, `temp_outside`, `gas_type`, `AC`, `rain`, `sun`

# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
print()
from IPython.display import display

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error # 回帰問題における性能評価に関する関数
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix # 回帰問題における性能評価に関する関数
from sklearn.model_selection import train_test_split # ホールドアウト法に関する関数
from sklearn.model_selection import KFold # 交差検証法に関する関数
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.svm import SVR

# 実行に影響のないWarningを非表示にする
import warnings
warnings.filterwarnings('ignore')


# # 1. データの読み込み・整形
# ## 1.1 データの読み込み

# In[28]:


# csvファイルの読み込み
# 小数点が”ピリオド”ではなく”カンマ（ドイツ式）のため、decimalを指定
df_raw = pd.read_csv("../../../input/anderas_car-consume/measurements.csv", decimal=',')
print(df_raw.head())


# ## 1.2 データの整形
# ### 1.2.1 不要データの削除
# - `special`：AC, rain, sunに「0or1]で記載があるので無視
# - `refill liters`, `refill gas`：タンク内ガス残量とかも関係しそうだけど、いったん無視  
# 
# ### 1.2.2 ダミー変数化
# - `gas_type`：(E10, SP98) = (1, 0)として、ダミー変数化  
# 
# ### 1.2.3 欠損地の処理
# - `temp_inside`に欠損値あり
# - 時系列データと解釈し、欠損直前データで穴埋め 

# In[29]:


# 必要データの取り出し
df_fuel = df_raw[['consume', 'distance', 'speed', 'temp_inside', 'temp_outside', 'gas_type', 'AC', 'rain', 'sun']]            .replace({'E10': 1, 'SP98': 0})            .fillna(method='ffill')            .rename(columns={ 'temp_inside':'temp_in','temp_outside':'temp_out'})
print(df_fuel.head())


# # 2. データの可視化

# ## 2.1 散布図行列
# - 量的変数について図示
# - カテゴリカル変数にて分類表示も可能

# In[30]:


sns.set()
print()


# ## 2.2 ヒートマップ

# In[31]:


# 相関係数をヒートマップにして可視化
fig = plt.figure(figsize=(8,6))
print()
print()


# ## 2.2 箱ひげ図
# - カテゴリカル変数を図示

# In[32]:


# seaborn
sns.set()
sns.set_style('whitegrid')
sns.set_palette('Dark2')
fig = plt.figure(figsize=(16,8))

# gas_type
ax = fig.add_subplot(1,4,1)
sns.boxplot(x='gas_type', y='consume', data=df_fuel, showfliers=False, ax=ax)
sns.stripplot(x='gas_type', y='consume', data=df_fuel, jitter=True, ax=ax)
# AC
ax = fig.add_subplot(1,4,2)
sns.boxplot(x='AC', y='consume', data=df_fuel, showfliers=False, ax=ax)
sns.stripplot(x='AC', y='consume', data=df_fuel, jitter=True, ax=ax)
# sun
ax = fig.add_subplot(1,4,3)
sns.boxplot(x='sun', y='consume', data=df_fuel, showfliers=False, ax=ax)
sns.stripplot(x='sun', y='consume', data=df_fuel, jitter=True, ax=ax)
# rain
ax = fig.add_subplot(1,4,4)
sns.boxplot(x='rain', y='consume', data=df_fuel, showfliers=False, ax=ax)
sns.stripplot(x='rain', y='consume', data=df_fuel, jitter=True, ax=ax)

print()


# # 3. 学習

# In[33]:


#精度格納用
df_precision = pd.DataFrame(index=['MSE','RMSE','MAE','RMSE/MAE','R2_fit','R2_pred'])
print(df_precision)


# ## 3.1 データを学習用・テスト用に分離

# In[34]:


# sk-learin, train_test_split を利用
X = df_fuel.drop(columns='consume').values
y = df_fuel['consume'].values.reshape(-1,1) #scikit-learn入力用に整形
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)


# ## 3.2 データの標準化

# In[35]:


# Graph Title を格納
titles = np.array(['distanc','speed','temp_in','temp_out','gas_type','AC','rain','sun'])

sns.set()
fig = plt.figure(figsize=(16,6))
for i in range(8):
    ax = fig.add_subplot(2, 4, i+1)
    plt.hist(X_train[:, i])
    plt.title(titles[i])
plt.tight_layout()


# ### `distance`が対数正規分布っぽい ⇒logを取ろう！
#     対数化しないほうが精度良かった

# In[36]:


# 対数化を行うか
#do_if = True
do_if = False

# Trueの場合に対数化を行う
if do_if == True:
    X_train[:,0] = np.log(X_train[:,0])
    X_test[:,0]  = np.log(X_test[:,0])
    #可視化
    sns.set()
    fig = plt.figure(figsize=(16,6))
    for i in range(8):
        ax = fig.add_subplot(2, 4, i+1)
        plt.hist(X_train[:, i])
        plt.title(titles[i])
    plt.tight_layout()


# ### `temp_in`を`temp_in - temp_out`にしてみよう！
#     温度差にしたほうが若干精度上がる

# In[37]:


# 車内の温度を車内外の温度差に置き換えるか？
do_if = True
#do_if = False

# Trueの場合に対数化を行う
if do_if == True:
    X_train[:,2] = X_train[:,2] - X_train[:,3]
    X_test[:,2]  = X_test[:,2]  - X_test[:,3]
    #可視化
    sns.set()
    fig = plt.figure(figsize=(16,6))
    for i in range(8):
        ax = fig.add_subplot(2, 4, i+1)
        plt.hist(X_train[:, i])
        plt.title(titles[i])
    plt.tight_layout()


# ### ようやく標準化

# In[38]:


# 標準化
stdsc = StandardScaler()
X_train_norm = X_train
X_test_norm = X_test
X_train_norm[:,:4] = stdsc.fit_transform(X_train[:,:4])
X_test_norm[:,:4] = stdsc.transform(X_test[:,:4])

# 可視化
sns.set()
fig = plt.figure(figsize=(16,6))
for i in range(8):
    ax = fig.add_subplot(2, 4, i+1)
    plt.hist(X_train[:, i])
    plt.title(titles[i])
plt.tight_layout()


# ## 3.3 線形回帰
# ### 3.3.1 プレーン

# In[39]:


# fit & predict
regr = LinearRegression()
regr.fit(X_train_norm, y_train)
y_pred = regr.predict(X_test_norm)

#係数の可視化
print('')
df_graph = pd.DataFrame()
df_graph['class'] = ['distance','speed','temp_in','temp_out','gas_type','AC','rain','sun']
df_graph['coef']  = regr.coef_.ravel()
sns.set()
sns.catplot(data=df_graph,x='class',y='coef',kind='bar',size=3, aspect=2.5)


# In[40]:


# 汎化誤差
# MSE, RMSE, MAE, RMSE/MAE, R2
mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred) 
score1 = regr.score(X_train_norm, y_train)
score2 = regr.score(X_test_norm, y_test)

# 精度を格納＆表示
df_precision['Linear'] =  [round(mse,3),round(rmse,3),round(mae,3),round(rmse/mae,3),round(score1,3),round(score2,3)]
print(df_precision)


# ### 3.2.2 正則化
# - 正則化手法として`ElasticNet`を試用
# - グリッドサーチ・交差検証法により最適なハイパーパラメータを計算

# In[41]:


# Grid Search
alpha_params = np.logspace(-3,1,5)
l1_ratio_params = np.arange(1, 11, 1)/10
param_grid = {'alpha': alpha_params, 'l1_ratio': l1_ratio_params}
clf = GridSearchCV(ElasticNet(), param_grid, cv=4, return_train_score=True, iid=False)
clf.fit(X_train_norm, y_train)

print('')
print(clf.best_params_)

#Scoreが最大となっているかの確認
df = pd.DataFrame(clf.cv_results_)[['param_alpha','param_l1_ratio','mean_test_score']]

sns.set()
sns.set_style('whitegrid')
fig = plt.figure(figsize=(8,4))
sns.pointplot(x='param_l1_ratio', y='mean_test_score', hue='param_alpha', data=df)


# In[42]:


# 最適パラメータを用いて識別する
clf2 = ElasticNet(**clf.best_params_)
clf2.fit(X_train_norm, y_train)
y_pred = clf2.predict(X_test_norm)

#係数の可視化
print('')
df_graph = pd.DataFrame()
df_graph['class'] = ['distance','speed','temp_in','temp_out','gas_type','AC','rain','sun']
df_graph['coef']  = regr.coef_.ravel()
sns.set()
sns.catplot(data=df_graph,x='class',y='coef',kind='bar',size=3, aspect=2.5)


# In[43]:


# 汎化誤差
# MSE, RMSE, MAE, RMSE/MAE, R2
mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred) 
score1 = clf.best_score_
score2 = clf2.score(X_test_norm, y_test)

# 精度を格納＆表示
df_precision['Linear(ElasticNet)'] =  [round(mse,3),round(rmse,3),round(mae,3),round(rmse/mae,3),round(score1,3),round(score2,3)]
print(df_precision)


# ## 3.3 SVR
# ### 3.3.1 線形カーネル

# In[44]:


# 学習
params_cnt = 10
param_C = np.logspace(0,1,params_cnt+1)
param_epsilon = np.logspace(-1,0,params_cnt+1)
params = params = { "C": param_C, "epsilon": param_epsilon }
clf = GridSearchCV(SVR(kernel="linear"), params, cv=4, scoring="r2", return_train_score=True, iid=False)
clf.fit(X_train_norm, y_train.ravel())
print('')


# In[45]:


# Best Score
print('')
for k, v in clf.best_params_.items():
    print( "log10("+k+") = %s"%round(np.log10(v),3))
          
# Scoreが最大かどうかの確認
df = pd.DataFrame(clf.cv_results_)[['param_C','param_epsilon','mean_test_score']]
df['param_C'] = np.round(np.log10(df['param_C'].values.tolist()), 3)
df['param_epsilon'] = np.round(np.log10(df['param_epsilon'].values.tolist()), 3)

sns.set()
sns.set_style('whitegrid')
fig = plt.figure(figsize=(8,4))
sns.pointplot(x='param_epsilon', y='mean_test_score', hue='param_C', data=df)


# In[46]:


# 最適パラメータを用いて識別する
clf2 = SVR(kernel="linear", **clf.best_params_)
clf2.fit(X_train_norm, y_train.ravel())
y_pred = clf2.predict(X_test_norm)


# In[47]:


# 汎化誤差
# MSE, RMSE, MAE, RMSE/MAE, R2
mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred) 
score1 = clf.best_score_
score2 = clf2.score(X_test_norm, y_test)

# 精度を格納＆表示
df_precision['SVR(Linear)'] =  [round(mse,3),round(rmse,3),round(mae,3),round(rmse/mae,3),round(score1,3),round(score2,3)]
print(df_precision)


# ### 3.3.1 非線形カーネル

# In[48]:


# 学習
params_cnt = 10
param_C = np.logspace(0,1,params_cnt+1)
param_epsilon = np.logspace(-2,0,params_cnt+1)
params =  { "C": param_C, "epsilon": param_epsilon}
clf = GridSearchCV(SVR(kernel='rbf',gamma='auto'), params, cv=4, scoring="r2", return_train_score=True ,iid=False)
clf.fit(X_train_norm, y_train.ravel())
print('')


# In[49]:


# Best Score
print('')
for k, v in clf.best_params_.items():
    print( "log10("+k+") = %s"%round(np.log10(v),3))

df = pd.DataFrame(clf.cv_results_)[['param_C','param_epsilon','mean_test_score']]
df['param_C'] = np.round(np.log10(df['param_C'].values.tolist()), 3)
df['param_epsilon'] = np.round(np.log10(df['param_epsilon'].values.tolist()), 3)

sns.set()
sns.set_style('whitegrid')
fig = plt.figure(figsize=(8,4))
sns.pointplot(x='param_epsilon', y='mean_test_score', hue='param_C', data=df)


# In[50]:


# 最適パラメータを用いて識別する
clf2 = SVR(kernel='rbf',gamma='auto',**clf.best_params_)
clf2.fit(X_train_norm, y_train.ravel())
y_pred = clf2.predict(X_test_norm)


# In[51]:


# 汎化誤差
# MSE, RMSE, MAE, RMSE/MAE, R2
mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred) 
score1 = clf.best_score_
score2 = clf2.score(X_test_norm, y_test)

# 精度を格納＆表示
df_precision['SVR(rbf)'] =  [round(mse,3),round(rmse,3),round(mae,3),round(rmse/mae,3),round(score1,3),round(score2,3)]
print(df_precision)


# ## 3.4 決定木

# In[52]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
import graphviz
#import pydotplus
from IPython.display import Image
from sklearn.externals.six import StringIO


# In[53]:


# 学習
max_depth = 10
param_depth = np.arange(1, max_depth+1)
params =  { "max_depth": param_depth}
clf = GridSearchCV(DecisionTreeRegressor(), params, cv=4, scoring="r2", return_train_score=True, iid=False)
clf.fit(X_train_norm, y_train.ravel())
print(clf.best_params_)

# 最適パラメータを用いて識別する
clf2 = DecisionTreeRegressor(**clf.best_params_)
clf2.fit(X_train_norm, y_train.ravel())
y_pred = clf2.predict(X_test_norm)


# In[54]:


# 汎化誤差
# MSE, RMSE, MAE, RMSE/MAE, R2
mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred) 
score1 = clf.best_score_
score2 = clf2.score(X_test_norm, y_test)

# 精度を格納＆表示
df_precision['DecisionTree'] =  [round(mse,3),round(rmse,3),round(mae,3),round(rmse/mae,3),round(score1,3),round(score2,3)]
print(df_precision)


# In[55]:


# 説明変数の重要度を出力する
# scikit-learnで算出される重要度は、ある説明変数による不純度の減少量合計である。
sns.set()
pd.DataFrame(clf2.feature_importances_, index=titles).plot.bar(figsize=(7,3))
plt.ylabel("Importance")
plt.xlabel("Features")
print()


# In[56]:


'''  dot_data = StringIO()  export_graphviz(clf2, out_file=dot_data, feature_names=['distance', 'speed', 'temp_inside', 'temp_outside', 'gas_type', 'AC', 'rain', 'sun'], filled=True, rounded=True, special_characters=True) graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) Image(graph.create_png()) ''' 


# ## 3.5 ランダム・フォレスト

# In[57]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import graphviz
#import pydotplus
from IPython.display import Image
from sklearn.externals.six import StringIO


# In[58]:


# 学習
regressor = RandomForestRegressor(min_samples_leaf=2, min_samples_split=2, random_state=1234, n_jobs=-1)
params = {'n_estimators': [3,5,10,100], 'max_depth': [2,3]}
clf = GridSearchCV( regressor, params, cv=4, scoring="r2", return_train_score=True, iid=False )
clf.fit(X_train_norm, y_train.ravel())
print(clf.best_params_)

# 最適パラメータを用いて識別する
clf2 = RandomForestRegressor(min_samples_leaf=2, min_samples_split=2, random_state=1234, n_jobs=-1,**clf.best_params_)
clf2.fit(X_train_norm, y_train.ravel())
y_pred = clf2.predict(X_test_norm)


# In[59]:


# 汎化誤差
# MSE, RMSE, MAE, RMSE/MAE, R2
mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred) 
score1 = clf.best_score_
score2 = clf2.score(X_test_norm, y_test)

# 精度を格納＆表示
df_precision['RandomForest'] =  [round(mse,3),round(rmse,3),round(mae,3),round(rmse/mae,3),round(score1,3),round(score2,3)]
print(df_precision)


# In[60]:


# 説明変数の重要度を出力する
# scikit-learnで算出される重要度は、ある説明変数による不純度の減少量合計である。
sns.set()
pd.DataFrame(clf2.feature_importances_, index=titles).plot.bar(figsize=(7,3))
plt.ylabel("Importance")
plt.xlabel("Features")
print()


# In[61]:


''' for i, est in enumerate(clf2.estimators_): if i==3: break print(i)   dot_data = StringIO()  export_graphviz(est, out_file=dot_data, feature_names=['distance', 'speed', 'temp_inside', 'temp_outside', 'gas_type', 'AC', 'rain', 'sun'], filled=True, rounded=True, special_characters=True) graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) display(Image(graph.create_png())) ''' 


# ## 3.6 アダブースト

# In[62]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


# In[63]:


# 学習
regressor = DecisionTreeRegressor( max_depth=3, min_samples_leaf=2, min_samples_split=2)
params = {'n_estimators': [3, 5, 10, 100]}
clf = GridSearchCV( AdaBoostRegressor( regressor,random_state=1234 ), params, cv=4, scoring="r2", return_train_score=True,iid=False ) 
clf.fit(X_train_norm, y_train.ravel())
print(clf.best_params_)

# 最適パラメータを用いて識別する
clf2 = AdaBoostRegressor( regressor,random_state=1234 ,**clf.best_params_)
clf2.fit(X_train_norm, y_train.ravel())
y_pred = clf2.predict(X_test_norm)


# In[64]:


# 汎化誤差
# MSE, RMSE, MAE, RMSE/MAE, R2
mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred) 
score1 = clf.best_score_
score2 = clf2.score(X_test_norm, y_test)

# 精度を格納＆表示
df_precision['Adaboost(Tree)'] =  [round(mse,3),round(rmse,3),round(mae,3),round(rmse/mae,3),round(score1,3),round(score2,3)]
print(df_precision)


# ## 3.6 ニューラルネットワーク
#     参考：https://qiita.com/cvusk/items/33867fbec742bda3f307

# In[65]:


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


# In[66]:


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


# In[67]:


print()


# In[68]:


# 汎化誤差
# MSE, RMSE, MAE, RMSE/MAE, R2
mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred) 
score1 = clf.best_score_
score2 = clf2.score(X_test_norm, y_test)

# 精度を格納＆表示
df_precision['NN'] =  [round(mse,3),round(rmse,3),round(mae,3),round(rmse/mae,3),round(score1,3),round(score2,3)]
print(df_precision)


# # Appendix. データの標準化・無相関化
# ##  無相関化

# In[69]:


data1 = df_fuel.drop('consume', axis=1).values
cov = np.cov(data1, rowvar=0) # 分散・共分散を求める
_, S = np.linalg.eig(cov)           # 分散共分散行列の固有ベクトルを用いて
data1_decorr = np.dot(S.T, data1.T).T #データを無相関化

fig = plt.figure(figsize=(8,6))
print()
print()

