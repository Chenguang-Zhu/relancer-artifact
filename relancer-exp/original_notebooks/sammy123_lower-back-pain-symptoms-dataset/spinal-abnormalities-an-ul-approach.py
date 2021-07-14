#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Engineer Nano Degree - Capstone Project
# ## Student: Nick Maiorana
# ## January 08, 2017
# 
# ## Overview
# 
# This project started as a work project that I performed for my professional career. The original project was used to identify false/positive readings from web sources that were marked as external attack points against our public web site. After completing the project successfully, my thoughts were that it would be perfect Capstone project for the Machine Learning Engineer Nanodegree program. It encompassed everything I had learned in the course and was applied in an actual environment with very accurate results. However, my employer is very wary of data (even anonymized data) from being used outside the company. My solution to this was to utilize the same process, however change the data source and goal of the project. For this I turned to a Kaggle dataset.
# 
# After searching the Kaggle available datasets, I decided to use one focused on lower back pain symptoms. This dataset offered me a real-world problem that needed solving, with enough information to exercise the process I used for my current job. I have always felt the medical community could benefit from either machine learning technologies or expert systems. This project will offer me the capability to see how accurate I can build a lower back diagnostic classifier, if all I had available to me were the data attributes.
# 
# The medical field is comprised of individual doctors and physicians that are tasked with diagnosing patient’s illnesses based on patient symptoms and test results. By using machine learning, these healthcare professionals can be assisted in their diagnosis by using patient data to predict the causes of the patient's symptoms.
# This project is designed to provide a methodology of taking raw data and transforming it into a classification tool that can be used to diagnose patients with normal or abnormal spinal conditions. The purpose of this project is to assume raw unclassified data is available and, through the process laid out here, derive classification for each data set through data grouping and random sampling.
# 
# A Kaggle dataset will be used for the feature and classification data. Although the data contains a classification attribute, this will only be used to identify clusters of data groups taken from random samples.
# The goal of this project is to generate a training data set using clustering techniques to separate records into Normal and Abnormal classes. Once the training data is created, a classification model is built and then scored against how well the classifier predicts the original classification of each record. 
# My goal is simple, determine how close to some of the Kaggle reports I can get a classifier to predict. The biggest difference is that I will not use the dataset directly, but will use unsupervised learning to determine classification attribute. In the end, I will use an accuracy score to compare my classifier against the actual dataset classification.
# 
# For reference, here are some Kaggle entries and their scores:
# 
# 
# __Kaggle Reference         | Score Method   | Score__
# ***
# - [Discriminating Normal-vs-Abnormal Spines](https://www.kaggle.com/antonio00/d/sammy123/lower-back-pain-symptoms-dataset/discriminating-normal-vs-abnormal-spines/discussion)  | Specificity | 54% 
# - [Using k-NN to Predict Abnormal Spines](https://www.kaggle.com/sammy123/lower-back-pain-symptoms-dataset/discussion/24230) | Accuracy | 93% 

# ## Process
# 
# The process used for this project is pretty straigt forward. It will take on the following steps:
# 
# - Load and display the original dataset
# - Remove and segregate classification information
# - Remove less important features
# - Scale and Transform the feature data
# - Use Unsupervised Learning to classify the data
# - Peform PCA to display UL classification information
# - Simulate cluster sample evaluation to identify cluster types (Normal/Abnormal)
# - Generate training dataset by classifying data using cluster analysis 
# - Determine a suitable classification model to be used
# - Generate model using final training dataset
# - Evaluate model classification against original classification attribute for accuracy

# ## Data source
# 
# The data for this project has been downloaded from Kaggle and is named the "Lower Back Pain Symptoms Dataset"  (https://www.kaggle.com/sammy123/lower-back-pain-symptoms-dataset) 
# 
# The information is layed out with 12 different featurs, labeld Col1-Col12 (Attribute1-12) and a classification attribute of "Abnormal" or "Normal". The descriptions for each attribute are as follows:
# 
# - Attribute1 = pelvic_incidence  (numeric) 
# - Attribute2 = pelvic_tilt (numeric) 
# - Attribute3 = lumbar_lordosis_angle (numeric) 
# - Attribute4 = sacral_slope (numeric) 
# - Attribute5 = pelvic_radius (numeric) 
# - Attribute6 = degree_spondylolisthesis (numeric) 
# - Attribute7 = pelvic_slope(numeric)
# - Attribute8 = direct_tilt(numeric)
# - Attribute9 = thoracic_slope(numeric)
# - Attribute10 = cervical_tilt(numeric)
# - Attribute11 = sacrum_angle(numeric)
# - Attribute12 = scoliosis_slope(numeric)
#  
# Although not necessary, for this project the feature names will be replaced with their descriptions to allow for easier feature analsyis. 
# 
# The classification attribute will be used to classify clusters or groups of data items with similar features and will not be directly used in constructing a classifier for the dataset. The origninal classificatin attribute will be used to evaluate the final classification model.

# ## Data Gathering
# ### Data Import and Display

# In[ ]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd

from IPython.display import display # Allows the use of display() for DataFrames

# Show matplotlib plots inline (nicely formatted in the notebook)

input_file = "../../../input/sammy123_lower-back-pain-symptoms-dataset/Dataset_spine.csv"
input_data = pd.read_csv(input_file)

renamed_columns = { 'Col1': 'pelvic_incidence', 'Col2': 'pelvic_tilt', 'Col3': 'lumbar_lordosis_angle', 'Col4': 'sacral_slope', 'Col5': 'pelvic_radius', 'Col6': 'degree_spondylolisthesis', 'Col7': 'pelvic_slope', 'Col8': 'direct_tilt', 'Col9': 'thoracic_slope', 'Col10': 'cervical_tilt', 'Col11': 'sacrum_angle', 'Col12': 'scoliosis_slope', 'Class_att' : 'classification' } 
input_data.rename(columns=renamed_columns, inplace=True)
input_data.drop(input_data.columns[13], axis=1, inplace=True)
print(input_data.head())
print(input_data.tail())
print(input_data.describe())


# #### Sample and Remove non-feature data
# 
# Since this dataset only has 310 items, the entire dataset will be used for this project. The ability to reduce the data by sampling a subset of the information is provided via the 'sample_size' variable and can be used if the dataset grows to an unmanagible amount. 
# 
# As can be seen below, the data set is unbalanced. There are 210 Abnormal spinal data points and only 100 Normal spinal datapoints.
# 
# I also removed the classification attribute since it is not needed for the initial phase of this project. The intent is to use a clustering algorythm to group the data into common subgroups. This classification attribute will be used later during the cluster analysis to identify how each cluster should be classified. It will also be used as a final step to determine the accuracy of the classificaiton model generated.

# In[ ]:


# take a sample size of the data, 1.0 if the data is not too large.
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

sample_size = 1.0
data = input_data.sample(frac=sample_size)

# Randomize the data
data = data.sample(frac=1.0).reset_index(drop=True)

print(data.head())

# Extract the classification column from the data
classification = data[['classification']]
print(classification.describe())
print(classification.head())
classification.describe().to_csv("classification_stats.csv", float_format='%.6f', index=True)


# ## Feature Relevance
# 
# ### Feature Relevance Part 1
# 
# In this step we will determine how relavent each feature is to determining spinal conditions for patients. To do this we will run through the dataset 12 times, and each time remove a feature. For each iteration, we will create and fit a regressor model to the subset of data so see how accurate the subset can predict the value of that removed feature. 
# 
# For starters, I will put any feature that scores less than 0.0 as the suggested features to drop
# 
# This will give us an indication of which features to keep as we build our classificaiton model.

# In[ ]:


import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing

# Scale the data using the natural logarithm
#preprocessed_data = reduced_feature_data.apply(np.log)

# Scale the data using preprocessing function in sklearn
feature_cols = list(data.columns[:-1])
preprocessed_data = data.copy()

suggested_features_to_drop = []
for feature in feature_cols:
    # Make a copy of the DataFrame, using the 'drop' function to drop the given feature
    # Extract the values of the dataframe to be used for the regression
    new_data = preprocessed_data.drop([feature], axis = 1, inplace = False)
    remaining_cols = list(new_data.columns[:-1])
    new_data_values = new_data[remaining_cols].values
    target_label = data[feature].values

    # Split the data into training and testing sets using the given feature as the target
    X_train, X_test, y_train, y_test = train_test_split(new_data_values, target_label, test_size=0.20, random_state=42)

    # Create a decision tree regressor and fit it to the training set
    regressor = tree.DecisionTreeRegressor(random_state=42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    # Report the score of the prediction using the testing set
    score = regressor.score(X_test, y_test)
    print (feature, score)
    if score < 0.0:
        suggested_features_to_drop.append(feature)
        

print ("\nSuggested features to drop:")
for feature in suggested_features_to_drop: print (feature)


# ### Feature Relevance Part 2
# 
# By reviewing the scores above, I will decide on which features to remove. I may revisit this to remove additional features as the analysis continues. In the segment below, I will indicate the initial features removed as well as any additional features as I progress through the project via comments. The goal is to reduce the feature set to a managable number of dimensions with at least 95% varience.
# 
# For my initial feature removal, I will eliminate the features with negative scores.

# In[ ]:


import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Initial set of features to drop by using the features with negative scores above
dropped_features = ['pelvic_radius', 'pelvic_slope', 'direct_tilt', 'thoracic_slope', 'cervical_tilt', 'sacrum_angle', 'scoliosis_slope']
 
reduced_feature_data = data.copy()
for feature in dropped_features:
    print ("Manually dropping feature:", feature)
    reduced_feature_data.drop([feature], axis = 1, inplace = True)
    
print(reduced_feature_data.head())
reduced_feature_data.to_csv("reduced_feature_data.csv", float_format='%.6f', index=False)


# ## Feature Scaling & Transformation
# 
# ### Feature Scaling
# 
# - In order to create a greater amount of variance, we will adjust the values for all the features by e^x.
# - To verify we are getting a better distribution of data, we will visualize the data befor and after scaling.
# - The scatter matrix below will indicate which features will work well together by showing the greatest amount of variance. This can also be used to return to the last section to remove additional features from the data. What we are looking for are gaussian distributions of the data to gain the most variance between the features.
# - Using the logarithmic function for scaling certain values may generate NA values. We will remove the NA values from the dataset.
# - The scaled dataset will be stored so that it can be used later without having to recreate it from start. This is done throught the project to allow for quick experimentation.

# #### Visualize the original data

# In[ ]:


import pandas as pd

reduced_feature_data = pd.read_csv("../../../input/sammy123_lower-back-pain-symptoms-dataset/Dataset_spine.csv")
feature_cols = list(reduced_feature_data.columns[:-1])

# Visulalize the original data
pd.scatter_matrix(reduced_feature_data[feature_cols], alpha = 0.3, figsize = (8,10), diagonal = 'kde');


# #### Scale the data
# 
# Here I will apply a function to scale the data to make it more linearly separable and create more variance. The log function I used generated some NA values in the data, which had to be removed. It is important to review the amount of data remaining since a significant percentage drop may leave the remaining data useless for further diagnosis.
# 
# I also provided an alternative to use sklearn.

# In[ ]:


import pandas as pd
from sklearn import preprocessing
import numpy as np

feature_cols = list(reduced_feature_data.columns[:-1])
target_col = reduced_feature_data.columns[-1] 
preprocessed_data = reduced_feature_data.copy()

# Scale the data using the natural logarithm

if True:
    preprocessed_data = preprocessed_data[feature_cols].apply(np.log)
    preprocessed_data = pd.concat([preprocessed_data, reduced_feature_data[target_col]], axis = 1)

# Scale the data using preprocessing function in sklearn

if False:
    scaler=preprocessing.StandardScaler()
    scaler.fit(reduced_feature_data[feature_cols])
    preprocessed_data[feature_cols] = scaler.transform(reduced_feature_data[feature_cols])

    print(preprocessed_data.describe())

# Drop any rows that contain NA values due to the scaling
preprocessed_data.dropna(subset=feature_cols, inplace=True)

# Display the percentage of the dataset kept.
percent_kept = (float(len(preprocessed_data)) / float(len(reduced_feature_data))) * 100.00
print ("Percentage kept:  {:2.2f}%".format(percent_kept))


# #### Store and display scaled data
# 
# Here we will re-introduce the classification data (to be used later during cluster sampling analysis) and persist the data in a file. This allows us to experiment throughout the project w/out having to start from the beginning every time. 

# In[ ]:


import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Output data set so that the analysis can be restarted w/out having to replay the entire notebook
preprocessed_data.to_csv("scaled_data.csv", float_format='%.6f', index=False)

# Visualize the scaled data
print(preprocessed_data.head())
pd.scatter_matrix(preprocessed_data, alpha = 0.3, figsize = (8,10), diagonal = 'kde');


# Outlier Detection
# 
# In this step I will identify (if any) any feature containing outliers using [Tukey's Method for identfying outliers](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/). If any feature value is determined to be an outlier, that data point will be removed from the dataset.

# In[ ]:


import pandas as pd
import numpy as np
from collections import Counter
from IPython.display import display # Allows the use of display() for DataFrames

# Get the scaled data data set so that the analysis can be restarted w/out having to replay the entire notebook
scaled_data = pd.read_csv("../../../input/sammy123_lower-back-pain-symptoms-dataset/Dataset_spine.csv")

# For each feature find the data points with extreme high or low values
outlierCounts = np.array([])

feature_cols = list(scaled_data.columns[:-1])

for feature in feature_cols:
    
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(scaled_data[feature].values, 25)
    
    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(scaled_data[feature].values, 75)
    
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3 - Q1)
    
    # Display the outliers
    print ("Data points considered outliers for the feature '{}':".format(feature))
    is_outlier = ~((scaled_data[feature] >= Q1 - step) & (scaled_data[feature] <= Q3 + step))
    print(scaled_data[~((scaled_data[feature] >= Q1 - step) & (scaled_data[feature] <= Q3 + step))])
    outlierCounts = np.append(outlierCounts, scaled_data[~((scaled_data[feature] >= Q1 - step) & (scaled_data[feature] <= Q3 + step))].index)

# OPTIONAL: Select the indices for data points you wish to remove
outlierCounts = outlierCounts.astype(int)
outlierCounted = Counter(outlierCounts)

print ("Number of data points that have one or more outlier features: ", len(outlierCounted))

outliers = [key for key,val in outlierCounted.items()]
print ("Data points with outliers: ", outliers)
# Remove the outliers, if any were specified
good_scaled_data = scaled_data.drop(scaled_data.index[outliers]).reset_index(drop = True)

print(good_scaled_data.describe())

# Output good scaled data set so that the analysis can be restarted w/out having to replay the entire notebook
good_scaled_data.to_csv("good_scaled_data.csv", float_format='%.6f', index=False)


# #### Use Clustering to group data 
# 
# For this stage of the analysis I decided to use K-Means clustering. The clustering portion is to isolate each classificaiton (Normal/Abnormal) into distinct clusters so that a sampling from each cluster will allow me to identify what type of data is in each cluster. The goal is to get to a point where each cluster is mostly homogenous (90%) so that I can use the cluster information to classify the original data.
# 
# This may involve repeating this process so that the clustering function can be accuratly tuned. In other words, once the clustering is complete, sample data from each cluster will be taken and scored for homogeniality. If the desired 90% is not achieved, the function will be tuned and retried.
# 
# - Attempt to use a range of cluster components
# - Score each one
# - Use the one with the best score

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

# Get the scaled data data set so that the analysis can be restarted w/out having to replay the entire notebook
good_scaled_data = pd.read_csv("../../../input/sammy123_lower-back-pain-symptoms-dataset/Dataset_spine.csv")

feature_cols = list(good_scaled_data.columns[:-1])
feature_data = good_scaled_data[feature_cols]

kmeans = KMeans(n_clusters=3)
afprop =  AffinityPropagation(preference=-50)
gmm_spherical = GaussianMixture(n_components=3, covariance_type='spherical', random_state = 42)

clusterers = [kmeans, afprop, gmm_spherical]

best = -1.0
for clusterer in clusterers:
    clusterer.fit(feature_data)

    # Predict the cluster for each data point
    preds = clusterer.predict(feature_data)

    # Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(feature_data, preds)
    
    print ("Clusterer {}      score {:0.4f}.".format(type(clusterer).__name__, score))

    if best < score:
        best = score
        best_clusterer = clusterer
        best_preds = preds

print ("\nThe best score was {:0.4f} using {} clusterer.".format(best, type(best_clusterer).__name__))

predictions = pd.DataFrame(best_preds, columns = ['cluster'])
predictions.to_csv("predictions.csv", float_format='%.6f', index=False)

print ("\n", predictions.groupby(['cluster']).size())

print ("\nClustering complete.")


# #### Display the Cluster
# 
# I will use PCA to perform feature reduction to present a graph of the cluster results

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from IPython.display import display # Allows the use of display() for DataFrames


good_scaled_data = pd.read_csv("../../../input/sammy123_lower-back-pain-symptoms-dataset/Dataset_spine.csv")
predictions = pd.read_csv("../../../input/sammy123_lower-back-pain-symptoms-dataset/Dataset_spine.csv")
feature_data = good_scaled_data.drop(['classification'], axis = 1)

pca_components = 2
pca = PCA(n_components=pca_components).fit(feature_data)
reduced_data = pca.transform(feature_data)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

# Output data set
reduced_data.to_csv("reduced_data.csv", float_format='%.6f', index=False)
plot_data = pd.concat([predictions, reduced_data], axis = 1)

# Generate the cluster plot
fig, ax = plt.subplots(figsize = (8,8))

# Color map
cmap = cm.get_cmap('gist_rainbow')

clusters = plot_data['cluster'].unique()

# Color the points based on assigned cluster
for i, cluster in plot_data.groupby('cluster'):   
    cluster.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2',                  color = cmap((i)*1.0/(len(clusters)-1)), label = 'Cluster %i'%(i), s=200);

# Set plot title
ax.set_title("Cluster Learning on PCA-Reduced Data");


# #### Recombine scaled data with cluster information
# 
# - Here we can map back in the descriptive data (cluster, Dimension 1, Dimesion 2, ... , Dimension N) so that we can begin to analyze what kind of data is in each cluster.
# - Save the full data set along with cluster information for further analysis

# In[ ]:


import pandas as pd

good_scaled_data = pd.read_csv("../../../input/sammy123_lower-back-pain-symptoms-dataset/Dataset_spine.csv")
reduced_data = pd.read_csv("../../../input/sammy123_lower-back-pain-symptoms-dataset/Dataset_spine.csv")
predicitons = pd.read_csv("../../../input/sammy123_lower-back-pain-symptoms-dataset/Dataset_spine.csv")

full_data = pd.concat([good_scaled_data, predictions, reduced_data], axis = 1)

full_data.to_csv('full_processed_data.csv', index=False)
print(full_data.head())


# #### Gather Data Samples
# 
# Here I will gather a sampling of data to analyze how pure each identified cluster is. In other words, I'm looking for cluster samples with mostly all of the same class (Normal or Abnormal). Under real-world circumstances, this would be a manual step to research the sample set to identify which class it belongs to. For the purposes of this project, the original classification attribute will be used to simulate this analysis.
# 
# I will be looking to sample a humanly manageable amount of data. The current data set has approximately 250 items. A manageable number would be approximately 20 items.
# 
# #### Store sample data for analysis
# 
# Under normal circumstances, the classification information would not be available to us. At this step in the process, we would need to identify the sample data to see which kind of data is in each cluster. This will allow us to generate our training data set from our original data set (w/out having classification). 
# 
# The importance of this is that it allow us to use data samples instead of the entire dataset to provide a classfication for each of the data items. In a normal dataset with 1000s or more data items, it would be humanly impossible to classify each item. By using the information derived from our clustering work, we can generalize the clusters into the appropropriate classification for that cluster.
# 
# My proposal for a real-world scenario is to store the sample data gathered above, and perform the appropriate analysis to classify each sample. Once this is done, we can reasonably assign a classification to each cluster. The analyzed date can be read into memory to begin constructing the final training set. 
# 
# For this project, we will use the conveniently provided classification information to determine how each cluster should be classified.

# In[ ]:


import pandas as pd

full_data = pd.read_csv("../../../input/sammy123_lower-back-pain-symptoms-dataset/Dataset_spine.csv")

sample_data = full_data.sample(frac=0.0)
frac_samples_per_cluster = 0.20 / float(len(full_data['cluster'].unique()))
samples_per_cluster = 20

clusters = full_data['cluster'].unique()

for cluster_number in clusters:
    cluster_data = full_data[full_data['cluster'] == cluster_number].sample(n=samples_per_cluster)
    sample_data = sample_data.append(cluster_data)

print ("Chosen samples segments dataset:")
#sample_data = full_data.sample(n=20)
print(sample_data.sort_values(by=['cluster']))

sample_data.to_csv('sample_data.csv', index=False)


# #### Analyze Samples
# 
# Here is where the manually analysed samples can be used to identify clusters.
# 
# - Determine how pure each cluster sample is for individual classification (Normal/Abnormal)
# - If the cluster samples are too mixed, consider using additional cluster components to further isolate each class

# In[ ]:


import pandas as pd
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics import adjusted_rand_score

sample_data = pd.read_csv("../../../input/sammy123_lower-back-pain-symptoms-dataset/Dataset_spine.csv")

print("Homogeneity Score is:       %.2f" % homogeneity_score(sample_data['cluster'].values, sample_data['classification'].values))
print("V Measure Score is:         %.2f" % v_measure_score(sample_data['cluster'].values, sample_data['classification'].values))
print("Adjusted Rand Score is:     %.2f" % adjusted_rand_score(sample_data['cluster'].values, sample_data['classification'].values))


# ### Determine which clusters contain "Abnormal" data points

# In[ ]:


import pandas as pd
from scipy.stats import mode

abnormal_clusters = []
sample_data = pd.read_csv("../../../input/sammy123_lower-back-pain-symptoms-dataset/Dataset_spine.csv")
clusters = sample_data['cluster'].unique()

for cluster_number in clusters:
    cluster_data = sample_data[sample_data['cluster'] == cluster_number]
    
    cluster_size = len(cluster_data)
    print ("Cluster", cluster_number, "has", cluster_size, "elements")
    
    cluster_classification = mode(cluster_data[['classification']])[0][0][0]
    
    print ("Cluster", cluster_number, "will be classified as", cluster_classification)
    
    if cluster_classification == 'Abnormal':
        abnormal_clusters.append(cluster_number)


abnormal_preds = full_data[(full_data['cluster'].isin(abnormal_clusters))]
normal_preds = full_data[(~full_data['cluster'].isin(abnormal_clusters))]
print ("Abnormal Preds:            {}".format(len(abnormal_preds.index)))
print ("Normal Preds:              {}".format(len(normal_preds.index)))


# #### Visualize the cluster
# 
# This step is not necessary, but will present the cluster analayis over the first 2 dimensions and samples visually to provide a graphical representation of our data.
# 
# - Blue ^ are Normal data points
# - Black v are Abnormal data points

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from IPython.display import display # Allows the use of display() for DataFrames


# Visualize the cluster data
full_data = pd.read_csv("../../../input/sammy123_lower-back-pain-symptoms-dataset/Dataset_spine.csv")
sample_data = pd.read_csv("../../../input/sammy123_lower-back-pain-symptoms-dataset/Dataset_spine.csv")

pca_samples = sample_data[['Dimension 1', 'Dimension 2', 'classification']]

plot_data = full_data[['cluster', 'Dimension 1', 'Dimension 2']]
# Generate the cluster plot
fig, ax = plt.subplots(figsize = (8,8))

# Color map
cmap = cm.get_cmap('gist_rainbow')

centers = plot_data['cluster'].unique()

# Color the points based on assigned cluster
for i, cluster in plot_data.groupby('cluster'):   
    cluster.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2',                  color = cmap((i)*1.0/(len(centers)-1)), label = 'Cluster %i'%(i), s=200);

# Plot transformed abnormal sample points 
abnormal_samples = pca_samples[(pca_samples['classification'] == 'Abnormal')].values
ax.scatter(x = abnormal_samples[:,0], y = abnormal_samples[:,1],            s = 30, linewidth = 1, color = 'black', marker = 'v');

# Plot transformed normal sample points 
abnormal_samples = pca_samples[(pca_samples['classification'] == 'Normal')].values
ax.scatter(x = abnormal_samples[:,0], y = abnormal_samples[:,1],            s = 30, linewidth = 1, color = 'blue', marker = '^');
# Set plot title
ax.set_title("Cluster Learning on PCA-Reduced Data - Sample Classifcations Marked [N = ^, A = v]");


# #### Build Training Dataset
# 
# - Build a training datafram using the full data and the cluster information.
# - Store the training data to a csv file

# In[ ]:


import pandas as pd
import numpy as np
from IPython.display import display # Allows the use of display() for DataFrames

full_data = pd.read_csv("../../../input/sammy123_lower-back-pain-symptoms-dataset/Dataset_spine.csv")
training_data = full_data.drop(['classification', 'Dimension 1', 'Dimension 2'], axis=1)

training_data.loc[training_data['cluster'].isin(abnormal_clusters), 'project_classification'] = 'Abnormal'
training_data.loc[~training_data['cluster'].isin(abnormal_clusters), 'project_classification'] = 'Normal'

training_data.drop(['cluster'], axis=1, inplace=True)

# Randomize the data
training_data = training_data.sample(frac=1.0)

training_data.to_csv('training_data.csv', float_format='%.6f', index=False)

print(training_data.head())
print ("Training data created.")


# ### Classification
# 
# In this section we will take the training data derived above to create a classifier

# #### Extract the Feature Data from the Target Data

# In[ ]:


import pandas as pd

training_data = pd.read_csv("../../../input/sammy123_lower-back-pain-symptoms-dataset/Dataset_spine.csv")

# Extract feature columns (first set)
feature_cols = list(training_data.columns[:-1])

# Extract target column 'false_positive' (last column)
target_col = training_data.columns[-1] 

# Show the list of columns
print ("Feature columns:\n{}".format(feature_cols))
print ("\nTarget column: {}".format(target_col))

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = training_data[feature_cols]
y_all = training_data[target_col]


# #### Train and Score Models
# 
# - We will attempt 3 different models
# - Train and score each one
# - Select the one with the best test scores

# In[ ]:


# Import the three supervised learning models from sklearn
import pandas as pd
from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score

# Initialize the three models
clf_A = tree.DecisionTreeClassifier(random_state=42)
clf_A_parameters = {'min_samples_split':(2, 200, 2000), 'min_samples_leaf': (2, 4, 6, 8 , 10), 'splitter': ('best', 'random')}

clf_B = SVC(random_state=42)
clf_B_parameters = {'kernel':('linear', 'rbf', 'sigmoid'), 'gamma': (1, 100, 1000, 10000, 100000), 'C':(1.0, 2.0, 3.0)}

clf_C = GaussianNB()
clf_C_parameters = {}

classifiers = [clf_A, clf_B, clf_C]
parameters = [clf_A_parameters, clf_B_parameters, clf_C_parameters]

# Loop through each classifier, train and test. Keep the one with the best test score
best_score = 0.0
for index in range(len(classifiers)):
    clf = classifiers[index]
    print ("\n{}:".format(clf.__class__.__name__))
    
    # Using Cross Validation
    
    scores = cross_val_score(clf, X_all, y_all, cv=10)
    print ("F1 mean score for CV training set: {:.4f}.".format(scores.mean()))

    if best_score < scores.mean():
        best_score = scores.mean()
        best_clf = clf
        best_parms = parameters[index]

print ("\nThe best testing score was : {:.4f}.".format(best_score))
print ("The best classifier was: {}".format(best_clf.__class__.__name__))
print ("The optional parameteres for this classifier was: {}".format(best_parms))


# #### Model Tuning
# 
# - Using GridSearch and Cross Validation, find optimal tuning parameters for classifier

# In[ ]:


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

# Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(best_clf, param_grid=best_parms, cv=10)

# Fit the grid search object to the training data and find the optimal parameters
grid_obj.fit(X_all, y_all)

print ("Best parameters set found on development set:")
print (grid_obj.best_params_)
print ()
# Get the estimator
clf = grid_obj.best_estimator_
print (clf)

print ("\nBest score: {:.4f}.".format(grid_obj.best_score_))


# #### Model Persitance
# 
# - Save the model to a file called classification_model.pkl
# - Test loading the model
# - Test the loaded model

# In[ ]:


from sklearn.externals import joblib
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split
import math

# Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test = train_test_split( X_all, y_all, train_size=0.75, test_size=0.25, random_state=42) 

# Show the results of the split
print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))

joblib.dump(clf, 'classification_model.pkl')
loaded_clf = joblib.load('classification_model.pkl')

testing_pred = loaded_clf.predict(X_test)
# Get Score based on training data
f1_score_testing_data = f1_score(y_test.values, testing_pred, pos_label="Abnormal")
print ("F1 score for test set: {:.4f}.".format(f1_score_testing_data))


# In[ ]:


# Finally score the original classification to the one derived by the model
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

full_data = pd.read_csv("../../../input/sammy123_lower-back-pain-symptoms-dataset/Dataset_spine.csv")
validation_data = full_data.drop(['cluster', 'Dimension 1', 'Dimension 2'], axis=1)

# Extract feature columns (first set)
feature_cols = list(validation_data.columns[:-1])

# Extract target column 'false_positive' (last column)
target_col = validation_data.columns[-1] 

# Show the list of columns
print ("Feature columns:\n{}".format(feature_cols))
print ("\nTarget column: {}".format(target_col))

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = validation_data[feature_cols]
y_all = validation_data[target_col]

loaded_clf = joblib.load('classification_model.pkl')

validation_pred = loaded_clf.predict(X_all)
# Get Score based on training data
validation_score = accuracy_score(y_all.values, validation_pred)
print ("\nValidation accuracy score: {:.4f}".format(validation_score))

target_names = ['Abnormal', 'Normal']

report = classification_report(y_all.values, validation_pred, target_names)
cf_matrix = confusion_matrix(y_all.values, validation_pred, target_names)

confusion = pd.crosstab(y_all, validation_pred, rownames=['Actual'], colnames=['Classified As'])

print(confusion)


cf_matrix = confusion.values
sensitivity = cf_matrix[0][0] / float(np.sum(cf_matrix[0][0:]))
specificity =  cf_matrix[1][1] / float(np.sum(cf_matrix[1][0:]))

print ("\nSensitivity:             {:.4f}".format(sensitivity))
print ("\nSpecificity:             {:.4f}".format(specificity))
print ("\nClassification Report\n", report)


# ### Conclusion
# 
# This was a very interesting data set to work with. I was very surprised that I could get to a 82% accuracy against the original classified data (88% of top Kaggle submission) using Unsupervised Learning and making conclusions via sampling. This is closer to how things work in the real-world. You don't always have the classification data readily available and you need to somehow separate the data points into similar buckets and make decisions on what your observations are about each bucket's makeup.
# 
# Comparing my results to those submitted by Kaggle members, I believe I did a good job considering I did not use the provided classification data to construct a classifier. One entry posted a 93% accuracy, 11% higher than the one I derived, but they had the benefit of using 100% of the provided classification data. In addition to the accuracy score, another entry posted a specificity score of 54%, while I could achieve a score of 86%.
# 
# I would have to say that the methodology used for this project can be applied to real-world problems where classification data is not available. In fact, I have used this same process for my job to classify false/positive transaction sources predicted by other systems.
# 
# I thoroughly enjoyed working this problem and getting much better in my Python and Pandas skill set.
# 
# 
