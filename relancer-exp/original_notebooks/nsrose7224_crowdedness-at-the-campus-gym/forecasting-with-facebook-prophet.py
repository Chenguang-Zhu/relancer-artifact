#!/usr/bin/env python
# coding: utf-8

# # I have decided to interpretate the results using Facebook recently realeased Prophet, for TimeSeries forecast

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
print()

df = pd.read_csv("../../../input/nsrose7224_crowdedness-at-the-campus-gym/data.csv")


# In[ ]:


from fbprophet import Prophet


# In[ ]:


df.head()


# In[ ]:


df.drop('timestamp', axis = 1, inplace = True)


# In[ ]:


df.head()


# In[ ]:


df['date'] = pd.to_datetime(df['date'])


# In[ ]:


df.info()


# ### Let's gonna group every hour , just to see what comes up

# In[ ]:


df = pd.DataFrame(df.set_index('date').groupby(pd.TimeGrouper('1H')).sum())


# In[ ]:


df.head()


# ### We have to check the number of 0s, because of the log problem

# In[ ]:


(df['number_people'] < 1).count()


# ## Ok, we have "13971 hours " when there was nobody at the gym . Maybe we can overcome this placing the inverse of log ? We'll see it.

# In[ ]:


#calcula o lucro da hora
df['Total'] = (df['number_people'])


# In[ ]:


df = df[['number_people']]


# In[ ]:


df.head()


# In[ ]:


df['ds'] = df.index
df.head()


# In[ ]:


df = df.rename(columns={'number_people': 'y'})


# In[ ]:


df['y'] = np.exp(df['y'])


# # Using Facebook Prophet 

# ## Facebook Prophet is a  tool for producing high quality forecasts for time series data that has multiple seasonality with linear or non-linear growth.
# 

# In[ ]:


from fbprophet import Prophet
m = Prophet(changepoint_prior_scale=0.001, mcmc_samples=500)
m.fit(df);


# In[ ]:


future = m.make_future_dataframe(periods=12000, freq='H')
future.tail()


# In[ ]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


import matplotlib.pyplot as plt
print()
m.plot(forecast);


# ### Finaly, we can see the best months, weeks and weekdays to attend the gym : 

# In[ ]:


m.plot_components(forecast);


# In[ ]:





# In[ ]:





# In[ ]:


import pandas as pd
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
print()
df = pd.read_csv("../../../input/nsrose7224_crowdedness-at-the-campus-gym/data.csv")


# [My code is on my github actually , as Kaggle does not accept some of the newest libraries I am using (Facebook Prophet) ][1]
# 
# 
#   [1]: https://github.com/andersonamaral/my_machine_learning_studies/blob/master/Crowdness_At_Gym_With_Prophet_Forecast.ipynb

# In[ ]:


## My code is on my github actually , as Kaggle does not accept some of the newest libraries I am using (Facebook Prophet)  :
#https://github.com/andersonamaral/my_machine_learning_studies/blob/master/Crowdness_At_Gym_With_Prophet_Forecast.ipynb


# In[ ]:


df.head()


# In[ ]:


df.drop('timestamp', axis = 1, inplace = True)


# In[ ]:


df['date'] = pd.to_datetime(df['date'])


# df = pd.DataFrame(df.set_index('date').groupby(pd.TimeGrouper('1H')).sum())

# In[ ]:


df = pd.DataFrame(df.set_index('date').groupby(pd.TimeGrouper('1H')).sum())


# In[ ]:


df['Total'] = (df['number_people'])


# In[ ]:


df = df[['number_people']]
df.head()


# In[ ]:


df['ds'] = df.index
df.head()


# In[ ]:


df = df.rename(columns={'number_people': 'y'})


# In[ ]:


df['y'] = np.exp(df['y'])


# In[ ]:


from fbprophet import Prophet
m = Prophet(changepoint_prior_scale=0.001, mcmc_samples=500)
m.fit(df);


# In[ ]:


future = m.make_future_dataframe(periods=12, freq='M')
future.tail()


# In[ ]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


import matplotlib.pyplot as plt
print()
m.plot(forecast);

