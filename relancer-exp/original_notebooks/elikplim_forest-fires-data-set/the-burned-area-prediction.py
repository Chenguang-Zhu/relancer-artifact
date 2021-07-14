#!/usr/bin/env python
# coding: utf-8

# In this reaserch i tried to make a prediction for the burned area within the Montesinho park. Forest Fires Data Set was used for this analysis. The data was clusterized. Stepwise regression methods were applied to choose one best predictor. It is interesting to see, which one of them has the biggest impact on the burned area in each cluster. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../../../input/elikplim_forest-fires-data-set"))

df = pd.read_csv("../../../input/elikplim_forest-fires-data-set/forestfires.csv")
df.head()


# In[ ]:


df_coordinates = df.loc[:, ["X", "Y"]]
coordinates = df_coordinates.values


# First, the coordinates were clusterized. The cluster amount was chosen using the elbow method

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0) 
    #max_iter - max number of iteration to define the final clusers
    #n_init - number of k_means algorithm running
    kmeans.fit(coordinates)
    wcss.append(kmeans.inertia_)
    #inertia_ Sum of squared distances of samples to their closest cluster center.
plt.plot(range(1, 20), wcss)
plt.title('Define the number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
print()


# There is last bend somewhere near fifth point, and then the curve is more smoothed. So, as i can see, the optimal number of clusters  is 5
# So, kmeans algithm with the same configurations was applied to find the clusters.

# In[ ]:


from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
plt.figure()
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0) 
clusters = kmeans.fit_predict(coordinates)
df["Cluster"]= clusters
plt.subplot()
plt.scatter(df['X'].values, df['Y'].values, marker='o', c=clusters, alpha=0.8)
plt.title("Clusters")
print()
centroids = kmeans.cluster_centers_
print(centroids)
print("\nCalculating distance between clusters\n")
print(euclidean_distances(centroids,centroids))


# And for each cluster, the burned area prediction was found
# ## Cluster 0

# In[ ]:


df_cluster0 = df[(df["Cluster"] == 0)] 
df_cluster0.head()


# In[ ]:


import seaborn as sns
import numpy as np
def build_cluster_corr(df_cluster):
    df_cluster_indicators = df_cluster.loc[:, ["area","FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]]
    plt.clf()
    plt.figure(figsize=(10,10))
    cmap = sns.diverging_palette(20, h_pos=220, s=75, l=50, sep=10, center='light', as_cmap=True)     
    corr_matrix = df_cluster_indicators.corr()
    corr_matrix[np.abs(corr_matrix) < 0.65] = 0
    print()
    print()


# In[ ]:


build_cluster_corr(df_cluster0)


# There is really small correlation values between data and dependent variable, so the stepwise regression methods was applied to chose the best predictors.

# In[ ]:


df_cluster0_indicators = df_cluster0.loc[:, ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind"]]
X = df_cluster0_indicators.values
Y = df_cluster0['area'].values
SL = 0.05
X_opt_ = X[:, [0, 1, 2, 3, 4, 5, 6]]


# In[ ]:


import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np
def forward_selection(x, y, sl):
    result = np.empty((len(X),1))
    numVars = len(x[0])
    all_regressors_OLS = smf.OLS(y, x).fit()
    maxVar = max(all_regressors_OLS.pvalues).astype(float)
    for i in range(0, numVars):
        regressor_OLS = smf.OLS(y, x[:,i]).fit()
        for j in range(0, numVars - i):
            p = regressor_OLS.pvalues[0].astype(float)
            if p > sl:
                if (p == maxVar):
                    result = np.insert(result, 0, j, axis=1)
                    
    plt.figure(figsize=(10,10))
    plt.scatter(result, y, color = 'red')
    plt.plot(result, regressor_OLS.predict(result), color = 'blue')
    plt.title('Forward Selection results')
    plt.xlabel('Predictor')
    plt.ylabel('area')
    print()
    print(regressor_OLS.summary())
    print(result)
    return result

X_Modeled_ = forward_selection(X_opt_, Y, SL)


# There is no predictor was chose by forward selection. I think the reason of that are unsignificant correlation values between the data, because the forward selection train the model using each predictor separately, so it is hard to choose really significant results. So the backward elimination algoritm was applied, to train the model using all predictors, and then choose the best one. 

# In[ ]:


def backward_elimination(x, y, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = smf.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    
    plt.figure(figsize=(10,10))
    plt.scatter(x, y, color = 'red')
    plt.plot(x, regressor_OLS.predict(x), color = 'blue')
    plt.title('Backward Elimination results')
    plt.xlabel('Predictor')
    plt.ylabel('area')
    print()
    print(regressor_OLS.summary())
    print(x)
    return x

X_Modeled_ = backward_elimination(X_opt_, Y, SL)


# The DMC predicor was chosen with r squared 14 %. 

# So, the best model for this cluster is area = 0.1324 * DMC

# ## Cluster 1

# In[ ]:


df_cluster1 = df[(df["Cluster"] == 1)] 
df_cluster1.head()


# In[ ]:


build_cluster_corr(df_cluster1)


# In[ ]:


df_cluster1_indicators = df_cluster1.loc[:, ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", 'rain']]
X = df_cluster1_indicators.values
Y = df_cluster1['area'].values
SL = 0.05
X_opt_ = X[:, [0, 1, 2, 3, 4, 5, 6, 7]]
X_Modeled_ = backward_elimination(X_opt_, Y, SL)


# The model for this cluster is: area = 0.2047 * DMC

# ## Cluster 2

# In[ ]:


df_cluster2 = df[(df["Cluster"] == 2)] 
df_cluster2.head()


# In[ ]:


build_cluster_corr(df_cluster2)


# In[ ]:


df_cluster2_indicators = df_cluster2.loc[:, ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", 'rain']]
X = df_cluster2_indicators.values
Y = df_cluster2['area'].values
SL = 0.05
X_opt_ = X[:, [0, 1, 2, 3, 4, 5, 6, 7]]
X_Modeled_ = backward_elimination(X_opt_, Y, SL)


# The model for this cluster is : area = 1.4330  * FFMC

# ## Cluster 3

# In[ ]:


df_cluster3 = df[(df["Cluster"] == 3)] 
df_cluster3.head()


# In[ ]:


build_cluster_corr(df_cluster3)


# In[ ]:


df_cluster3_indicators = df_cluster2.loc[:, ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind"]]
X = df_cluster3_indicators.values
Y = df_cluster3['area'].values
SL = 0.05
X_opt_ = X[:, [0, 1, 2, 3, 4, 5, 6]]
X_Modeled_ = backward_elimination(X_opt_, Y, SL)


# The  model for this cluster is : area = 1.777  * wind

# ## Cluster 4

# In[ ]:


df_cluster4 = df[(df["Cluster"] == 4)] 
df_cluster4.head()


# In[ ]:


build_cluster_corr(df_cluster4)


# In[ ]:


df_cluster4_indicators = df_cluster4.loc[:, ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", 'rain']]
X = df_cluster4_indicators.values
Y = df_cluster4['area'].values
SL = 0.05
X_opt_ = X[:, [0, 1, 2, 3, 4, 5, 6, 7]]
X_Modeled_ = backward_elimination(X_opt_, Y, SL)


# The best model for this cluster is backward elimination algoritm result: area = 0.9878  * temp
