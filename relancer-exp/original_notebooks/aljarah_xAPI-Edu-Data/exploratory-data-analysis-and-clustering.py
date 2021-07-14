#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../../../input/aljarah_xAPI-Edu-Data/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../../../input/aljarah_xAPI-Edu-Data"))

# Any results you write to the current directory are saved as output.


# In[ ]:



import numpy as np
import pandas as pd 


df = pd.read_csv("../../../input/aljarah_xAPI-Edu-Data/xAPI-Edu-Data.csv")

df.head(30)


# In[ ]:


print(df.shape)
df.isnull().sum()


# データは４８０行なのでだいぶ小さい。欠損値はなし。

# **データの可視化**
# 生徒のクラスとその他の特徴量の関係を見ていく。

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
#breakdown by class
sns.countplot(x="Topic", data=df, palette="muted")
print()


# ITやフランス語の授業が多い。

# In[ ]:


df['Failed'] = np.where(df['Class']=='L',1,0)
sns.factorplot('Topic','Failed',data=df,size=9)


# In[ ]:


pd.crosstab(df['Class'],df['Topic'])


# ITや化学でLowクラスになっている生徒が多く、地質学の授業では一人もいない。

# In[ ]:


sns.countplot(x='Class',data=df,palette='PuBu')
print()
df.Class.value_counts()


# In[ ]:


sns.countplot(x='ParentschoolSatisfaction',data = df, hue='Class',palette='bright')
print()


# 緒館通りだが、親の学校への満足度がよい場合ではLowクラスの子は少なく、満足度が低い場合はHighクラスの生徒は少ない。

# In[ ]:


sns.factorplot('Relation','Failed',data=df)


# 母親が子供の責任をとる場合のほうがLowクラスとなっている生徒の数が少ないようである。
# 統計的検定でこの差を確認していく。

# In[ ]:


from scipy.stats import ttest_ind
t, p = ttest_ind(df[df.Relation == "Father"].Failed, df[df.Relation == "Mum"].Failed)


# In[ ]:


print("t値,p値")
print(t,p)


# 十分p値は小さそうである。

# In[ ]:


sns.factorplot("gender","Failed",data=df)


# In[ ]:


t, p = ttest_ind(df[df.gender == "M"].Failed, df[df.gender == "F"].Failed)
print("t値,p値")
print(t,p)


# 性別に関して言えば男性のほうがlowクラス生徒が多いい用である。

# In[ ]:


Raised_hand = sns.boxplot(x="Class", y="raisedhands", data=df)
Raised_hand = sns.swarmplot(x="Class", y="raisedhands", data=df, color=".15")
print()


# In[ ]:


ax = sns.boxplot(x="Class", y="Discussion", data=df)
ax = sns.swarmplot(x="Class", y="Discussion", data=df, color=".25")
print()


# In[ ]:


Vis_res = sns.boxplot(x="Class", y="VisITedResources", data=df)
Vis_res = sns.swarmplot(x="Class", y="VisITedResources", data=df, color=".25")
print()


# In[ ]:


Anounce_bp = sns.boxplot(x="Class", y="AnnouncementsView", data=df)
Anounce_bp = sns.swarmplot(x="Class", y="AnnouncementsView", data=df, color=".25")
print() 


# 明らかにクラスごとに分布が違うようである。

# In[ ]:




# ４項間のペアプロットを見ると、やはり、クラスごとに特徴があるようだ。

# In[ ]:


Facetgrid = sns.FacetGrid(df,hue='Failed',size=6)
Facetgrid.map(sns.kdeplot,'raisedhands',shade=True)
Facetgrid.set(xlim=(0,df['raisedhands'].max()))
Facetgrid.add_legend()


# In[ ]:


Facetgrid = sns.FacetGrid(df,hue='Failed',size=7)
Facetgrid.map(sns.kdeplot,'Discussion',shade=True)
Facetgrid.set(xlim=(0,df['Discussion'].max()))
print()


# In[ ]:


Facetgrid = sns.FacetGrid(df,hue='Failed',size=7)
Facetgrid.map(sns.kdeplot,'VisITedResources',shade=True)
Facetgrid.set(xlim=(0,df['VisITedResources'].max()))
print()


# 授業の資料をみる生徒がレベルの高いクラスの生徒であるようだ。
# 

# また、三つの分布をみると成績が良い生徒にはっきりした特長は少ないが、成績が悪い生徒には明確に特徴がありそうである。

# In[ ]:





# In[ ]:


df2 = df.loc[:,["raisedhands","VisITedResources","AnnouncementsView","Discussion"]]


# In[ ]:


feature_names = ["raisedhands","VisITedResources","AnnouncementsView","Discussion"]


# In[ ]:


correlation_matrix = np.corrcoef(df2.transpose())


# In[ ]:




# 一応すべてに正の相関があったけど、手を上げることと、資料を見ることの相関が大きい。
# 資料をみて授業を理解しているから手を上げられるのかもしれない。

# In[ ]:


df.groupby('Topic').median()


# 1. lowクラスの多いITやMathで資料を見ている回数が少ないことがわかる。
# 1. どういうわけか、資料を読む回数が多いにもかかわらずChemistryはlowクラスの生徒が多い。
# 1. Spanish、Frenchといった語学科目はdiscussionが低いが科目の特性上、議論が難しいのだろう。Failed==0の確率分布が二つの山を持つ理由と思われる。

# In[ ]:


df['AbsBoolean'] = df['StudentAbsenceDays']
df['AbsBoolean'] = np.where(df['AbsBoolean'] == 'Under-7',0,1)
df['AbsBoolean'].groupby(df['Topic']).mean()


# 1. Geologyの生徒は欠席が少ない。Chemistryの生徒は欠席が多い。

# In[ ]:


df['TotalQ'] = df['Class']
df['TotalQ'].loc[df.TotalQ == 'L'] = 0.0
df['TotalQ'].loc[df.TotalQ == 'M'] = 1.0
df['TotalQ'].loc[df.TotalQ == 'H'] = 2.0




y = np.array(df['TotalQ'])


# In[ ]:


df.columns


# In[ ]:


categorical_features = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID','SectionID', 'Topic', 'Semester', 'Relation','ParentAnsweringSurvey', 'ParentschoolSatisfaction','StudentAbsenceDays']


# In[ ]:


categorical_features


# In[ ]:


X = pd.get_dummies(df.iloc[:,0:16],columns=categorical_features)


# In[ ]:


X


# k-meansクラスタリングによって新しい特徴量を作っていく。

# In[ ]:


from sklearn.cluster import KMeans
y_pred = pd.DataFrame()
for i in range(5):
    kmeans = KMeans(n_clusters = i+2)
    kmeans.fit(X)
    y_pred["cluster_"+str(i)] = kmeans.predict(X)


# In[ ]:


# ワンホットエンコーディングのためにカテゴリー化する
a =y_pred.astype(str)
a.dtypes


# In[ ]:


clustering = pd.get_dummies(a)


# In[ ]:


clustering


# In[ ]:


X_train = pd.concat([X,clustering],axis=1)


# In[ ]:


X_train = pd.concat([X_train,y_pred.iloc[:,4]],axis=1)


# In[ ]:


X_train['cluster_4'].astype(str)


# In[ ]:




# 訓練セットとテストセットに分ける。

# In[ ]:


from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler


X_train, X_test, y_train, y_test = train_test_split(X_train.drop('cluster_4',axis=1), y, test_size=0.3, random_state=0)




# **サポートベクタマシン　線形SVC　**

# In[ ]:


from sklearn.svm import SVC

svm = SVC(kernel='linear', C=2.0, random_state=0)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())


# In[ ]:


from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500,max_leaf_nodes=20,random_state=990)
rnd_clf.fit(X_train,y_train)
y_pred = rnd_clf.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())


# In[ ]:


from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


# In[ ]:


print(classification_report(y_test, y_pred))


# 適合率も再現率もF1値もほとんど同じくらいの値になった。
# 
# 次にランダムフォレストにより重要な特徴量を見ていく。

# In[ ]:


np.sort(rnd_clf.feature_importances_[:92])[::-1]


# In[ ]:


rnd_clf.n_features_


# In[ ]:


random_forest_feature = pd.DataFrame({"feature":X_train.columns,"importance":rnd_clf.feature_importances_})


# In[ ]:


random_forest_feature=random_forest_feature.sort_values(by = "importance",ascending=False)[:20]


# In[ ]:


random_forest_feature


# In[ ]:


random_forest_feature.loc[:,'importance'][::-1]


# In[ ]:


n_features = 20
plt.barh(range(20),random_forest_feature.loc[:,'importance'][::-1], align='center')
plt.yticks(np.arange(n_features),random_forest_feature.loc[:,'feature'][::-1])
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.ylim(-1, n_features)


# 結局のところ、ぎゅぎょうの資料をみているか、欠席の回数、授業で手を上げた回数、アナウンスを見るかどうかが重要であるようです。

# 

