#!/usr/bin/env python
# coding: utf-8

# **Kernel description:**
# 
# This kernel demonstrates the application of an [autoregressive model][1] to the problem of predicting avocado prices for various cities, states, and regions of the USA. This kernel was written as part of my university project on time series forecasting.
# 
# The dataset used is the [updated version][upd_dataset] of the [avocado dataset][original_dataset]. Please note that due to having lots of fluctuations in the data and the need to take the lag time span of at least 1 year for an AR model, almost all time series (except for the `Total U.S.` data) are quite tough ones to make reasonably accurate predictions for. For this reason, the `Total U.S.` data was used for the demonstration purposes.
# 
# There is not much information in this kernel but, still, please consider upvoting it if you liked it and/or got some insights from it!
# 
# PS. The Table of Contents was generated using ToC2 extension for Jupyter Notebook.
# 
# TODO:
#  * add stationarity tests
#  
#  [1]: https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python
#  [upd_dataset]: https://www.kaggle.com/timmate/avocado-prices-2020
#  [original_dataset]: https://www.kaggle.com/neuromusic/avocado-prices

# **Links**
# 
# Interesting and insightful kernels featuring other ML and DL methods:
# * https://www.kaggle.com/shahules/avocado-apocalypse
# * https://www.kaggle.com/ladylittlebee/linreg-knn-svr-decisiontreerandomforest-timeseries
# * https://www.kaggle.com/biphili/butter-fruit-avocado-price-forecast
# * https://www.kaggle.com/dimitreoliveira/deep-learning-for-time-series-forecasting
# * https://www.kaggle.com/dimitreoliveira/time-series-forecasting-with-lstm-autoencoders/input
# 
# Articles on autoregressive and ARIMA models:
# * https://towardsdatascience.com/machine-learning-part-19-time-series-and-autoregressive-integrated-moving-average-model-arima-c1005347b0d7
# * https://towardsdatascience.com/millennials-favorite-fruit-forecasting-avocado-prices-with-arima-models-5b46e4e0e914

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Read-the-dataset" data-toc-modified-id="Read-the-dataset-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Read the dataset</a></span></li><li><span><a href="#Preprocess-the-data" data-toc-modified-id="Preprocess-the-data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Preprocess the data</a></span></li><li><span><a href="#Get-a-subset-of-the-data-which-will-be-used-for-model-traning-and-making-predictions" data-toc-modified-id="Get-a-subset-of-the-data-which-will-be-used-for-model-traning-and-making-predictions-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Get a subset of the data which will be used for model traning and making predictions</a></span></li><li><span><a href="#Stationarize-the-subset" data-toc-modified-id="Stationarize-the-subset-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Stationarize the subset</a></span></li><li><span><a href="#Prepare-the-data-from-the-subset-for-the-model-training" data-toc-modified-id="Prepare-the-data-from-the-subset-for-the-model-training-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Prepare the data from the subset for the model training</a></span></li><li><span><a href="#Train-and-evaluate-the-AR-model" data-toc-modified-id="Train-and-evaluate-the-AR-model-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Train and evaluate the AR model</a></span></li><li><span><a href="#Plot-the-predictions-and-ground-truth-data" data-toc-modified-id="Plot-the-predictions-and-ground-truth-data-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Plot the predictions and ground-truth data</a></span></li></ul></div>

# In[ ]:


# This Python 3 environment comes with many helpfula analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../../../input/neuromusic_avocado-prices/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk("../../../input/neuromusic_avocado-prices"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set global parameters for plotting.
plt.rc('figure', figsize=(12, 6))
sns.set(font_scale=1.2)


# In[ ]:


import warnings

warnings.filterwarnings('ignore')


# ## Read the dataset

# Read the data from the dataset updated up to June 2020.

# In[ ]:


DATASET_PATH = "../../../input/neuromusic_avocado-prices/avocado.csv"

avocado_df = pd.read_csv(DATASET_PATH, parse_dates=['date'], index_col=['date']) 

avocado_df


# ## Preprocess the data

# Select only the columns that we need to perform TSA and average price forecasting using an AR model.

# In[ ]:


columns_considered = ['average_price', 'type', 'geography']
avocado_df = avocado_df[columns_considered]
avocado_df.head()


# Uncomment the lines below to print the number of entries for various cities, states, and regions.

# In[ ]:


# print('Number of entries for various cities and regions:')
# print()

# for geographical_name in avocado_df.geography.unique():
#     num_entries = sum(avocado_df.geography == geographical_name)
#     print(f'{geographical_name:25} {num_entries}')


# Plot the average price of conventional avocados in all regions over time (for each date, prices in all regions are plotted). 

# In[ ]:


sub_df = avocado_df.query("type == 'conventional'")

plt.scatter(sub_df.index, sub_df.average_price, cmap='plasma')
plt.title('Average price of conventional avocados in all regions and '           'cities over time')

plt.xlabel('Date')
plt.ylabel('Average price')
print()


# ## Get a subset of the data which will be used for model traning and making predictions

# In[ ]:


def plot_rolling_stats(time_series, window, avocado_type, geography):
    """ A helper function for plotting the given time series, its rolling mean and standard deviation. """ 

    rolling_mean = time_series.rolling(window=window).mean()
    rolling_std = time_series.rolling(window=window).std()

    index = time_series.index

    sns.lineplot(x=index, y=time_series.average_price, label='data', color='cornflowerblue') 
    
    sns.lineplot(x=index, y=rolling_mean.average_price, label='rolling mean', color='orange') 
    
    sns.lineplot(x=index, y=rolling_std.average_price, label='rolling std', color='seagreen') 
    
    plt.title(f'Average price of {avocado_type} avocados in {geography}')
    plt.xlabel('Date')
    plt.ylabel('Average price')    


# Choose a geography (i.e., a certain region, state, city, or the `Total U.S.` aggregated data) and an avocado type here. NB: `Total U.S.` contains the data which seems the most predictable in comparison to other geographical names of the U.S.

# In[ ]:


# NB: these two variables affect all the following calculations in that kernel.
AVOCADO_TYPE = 'conventional'
GEOGRAPHY = 'Total U.S.'

sub_df = avocado_df.query(f"type == '{AVOCADO_TYPE}' and "                           f"geography == '{GEOGRAPHY}'")
                          
sub_df.drop(['type', 'geography'], axis=1, inplace=True)
sub_df


# Resample the subset if needed (not really needed for the `Total U.S.` data). This leads to shrinking of the data, however, it might help to smoothen the data a bit and make it slighly easier to predict. 

# In[ ]:


# sub_df = sub_df.resample('2W').mean().bfill()
# sub_df.dropna(axis=0, inplace=True)
# sub_df


# Plot the chosen subset (time series), its rolling mean and standard deviation.

# In[ ]:


plot_rolling_stats(sub_df, window=4, avocado_type=AVOCADO_TYPE, geography=GEOGRAPHY) 


# ## Stationarize the subset

# Apply differencing of a given order (if needed).

# In[ ]:


# sub_df = sub_df.diff(periods=1)
# sub_df


# Differencing always results in at least one NaN value, so drop all NaNs appeared after the differencing.

# In[ ]:


# sub_df.dropna(axis=0, inplace=True)
# sub_df


# In[ ]:


# plot_rolling_stats(sub_df, window=4, avocado_type=AVOCADO_TYPE, region=REGION)


# ## Prepare the data from the subset for the model training

# Split the data into the training and test sets.

# In[ ]:


TEST_SET_SIZE = 45  # number of weeks left for the test set

data = sub_df.values
train_set, test_set = data[:-TEST_SET_SIZE], data[-TEST_SET_SIZE:]

print('shapes:', data.shape, train_set.shape, test_set.shape)


# Plot the training and test data.

# In[ ]:


train_set_size = len(data) - TEST_SET_SIZE
train_set_dates = sub_df.head(train_set_size).index  # for plotting
test_set_dates = sub_df.tail(TEST_SET_SIZE).index  

plt.plot(train_set_dates, train_set, color='cornflowerblue', label='train data')
plt.plot(test_set_dates, test_set, color='orange', label='test data')
plt.legend(loc='best')
plt.title(f'Average price of {AVOCADO_TYPE} avocados in {GEOGRAPHY}')
plt.xlabel('Date')
plt.ylabel('Average price')
print()


# ## Train and evaluate the AR model

# In[ ]:


from statsmodels.tsa.ar_model import AutoReg

model = AutoReg(train_set, lags=52)  # use time span of 1 year for lagging
trained_model = model.fit()
# print('Coefficients: %s' % trained_model.params)


# Get predictions and calculate an MSE and RMSE.

# In[ ]:


from sklearn.metrics import mean_squared_error as mse

predictions = trained_model.predict(start=train_set_size, end=train_set_size + TEST_SET_SIZE - 1) 

error = mse(test_set, predictions)

print(f'test MSE: {error:.3}')
print(f'test RMSE: {error ** 0.5:.3}')


# ## Plot the predictions and ground-truth data

# In[ ]:


plt.plot(test_set_dates, predictions, color='orange', label='predicted')
plt.plot(sub_df.index, sub_df.average_price, color='cornflowerblue', label='ground truth') 

plt.legend(loc='best')
plt.title(f'Average price of {AVOCADO_TYPE} avocados in {GEOGRAPHY}')
plt.xlabel('Date')
plt.ylabel('Average price')
print()


# In[ ]:




