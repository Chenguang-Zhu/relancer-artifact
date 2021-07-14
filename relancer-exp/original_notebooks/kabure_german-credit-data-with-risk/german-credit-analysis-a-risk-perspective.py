#!/usr/bin/env python
# coding: utf-8

# <img src="https://xixerone.com/wp-content/uploads/2017/11/D%C3%B3nde-dormir-en-Frankfurt-Mejores-zonas-y-hoteles.jpg">
# <h1 style="text-align:center;"> German Credit Analysis </h1>
# We will explore in this data analysis the behavior of German borrowers. Questions such as for what <b>purpose</b> do German borrowers apply for credit loans?,  How many jobs do each borrower have? , What patterns (if any) determine whether the loan will have a  good or bad risk ? Of course, many more questions will be answered through our in-depth analysis of German credit borrowers. To make our visualizations more interactive we will be using all our charts with plotly, an interactive library that will allow us to have a better insight about our data. <br>
# 
# *“Fools say that they learn by experience. I prefer to profit by others experience.” *- <b>Otto Von Bismarck</b>
# 
# <b>Note: This project will be subjected to constant updates in the near future. Also, I am open to listen insights from Kagglers about this project. </b> 
# 
# <h3> Goals for this Project </h3>
# <ul>
# <li> Explore our data and detecting <b>key patterns</b>. </li>
# <li> Develop a <b>Neural Network</b> to predict whether a loan will be of a good or bad risk. </li>
# <li> Most importantly, have <b>fun</b> while doing this project. </li>
# <ul>
# 

# <h2> Outline : </h2>
# 1) [Brief Overview](#brief_overview) <br><br>
# 
# 2) **Analysis by Type of Group**<br>
# a) [Gender Analysis](#gender_analysis) <br>
# b) [Age Groups](#age_groups)<br>
# c) [Wealth Analysis](#wealth_analysis)<br><br> 
# 
# 3) **Correlations and Purposes of Loans**<br>
# a) [Correlations](#correlations)<br>
# b) [Loan Purpose](#purpose_loans)<br>
# 
# 4) **Modeling**<br>
# a) [Predictive Modelling](#predictive_modelling)
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import tools
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import warnings
init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")

# Importing our data
df = pd.read_csv("../../../input/kabure_german-credit-data-with-risk/german_credit_data.csv",index_col=0)

original_df = df.copy()


# ## Brief Overview: 
# <a id="brief_overview"></a>
# The first phase of this project is to see what is our data made about. Which variables are numerical or categorical and which columns have "Null" values, which is something we will address in the feature engineering phase.
# 
# ## Summary:
# <ul>
# <li> We have four <b>numeric</b> and four <b>categorical</b> features. </li>
# <li> The <b>average age</b>  of people in our dataset is 35.54</li>
# <li> The <b>average credit amount</b> borrowed is 3271 </li>
# </ul>
# 
# 

# In[ ]:


df = df.rename(columns={"Credit amount": "Credit_amount"})


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


# Check missing values
df.isnull().sum().sort_values(ascending=False)


# In[ ]:


df.head()


# In[ ]:





# In[ ]:


# Distribution of Credit_Amount for each Gender
male_credit = df["Credit_amount"].loc[df["Sex"] == "male"].values
female_credit = df["Credit_amount"].loc[df["Sex"] == "male"].values
total_credit = df['Credit_amount'].values

fig, ax = plt.subplots(1, 3, figsize=(16,4))

sns.distplot(male_credit, ax=ax[0], color="#FE642E")
ax[0].set_title("Male Credit Distribution", fontsize=16)
sns.distplot(female_credit, ax=ax[1], color="#F781F3")
ax[1].set_title("Female Credit Distribution", fontsize=16)
sns.distplot(total_credit, ax=ax[2], color="#2E64FE")
ax[2].set_title("Total Credit Distribution", fontsize=16)
print()


# In[ ]:


df.head()


# In[ ]:


import seaborn as sns

plt.figure(figsize=(13,6)) #figure size
g = sns.boxplot(x='Purpose', y='Credit_amount', data=df, palette="RdBu") 


g.set_title("Credit Distribution by Purpose", fontsize=16)
g.set_xticklabels(g.get_xticklabels(),rotation=45) # It's the way to rotate the xticks when we use variable to our graphs
g.set_xlabel('Device Names', fontsize=18) # Xlabel
g.set_ylabel('Trans Revenue(log) Dist', fontsize=18) 
print()


# # Analysis by Group:
# 
# ## Gender Analysis:
# <a id="gender_analysis"></a>
# In this section analyze the gender section of our dataset.
# 
# ### Objectives: 
# <ul>
# <li> Find the distribution of genders in our dataset. </li>
# <li> See the distribution o each gender by the age (For instance, we have a higher number of young males than younger females) </li>
# <li> What were the main application reasons for a credit loan? Does it vary by Gender?</li>
# <li> How many jobs does each gender have? How many are Unemployed?</li>
# </ul>
# 
# ### Summary:
# <ul>
# <li>Theres <b>2x more males</b> than females in our dataset. </li>
# <li>Most females that applied for a credit loan were<b> less than 30</b> .</li>
# <li>Most of the males that applied for a loan ranged from their <b>20s-40s</b></li>
# <li>  Females were more likely to apply for a credit loan tobuy <b>furniture and equipment</b>. (10% more than males)</li>
# <li> Males applied 2x more than females for a credit loan to invest in a <b>business</b>. </li>
# <li> 2x of females were <b>unemployed</b> compared to males. </li>
# <li> 2x of males <b>worked 3 jobs</b> compared to females. </li>
# <li> Suprisingly, most people that applied for a credit loan have <b>two jobs</b>! </li>
# 
# </ul>
# 
# 

# In[ ]:


# We have 2x more German males applying for Credit Loans than Females.
df["Sex"].value_counts()


# In[ ]:


from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

by_age = df['Age'].values.tolist()
male_age = df['Age'].loc[df['Sex'] == 'male'].values.tolist()
female_age = df['Age'].loc[df['Sex'] == 'female'].values.tolist()

trace0 = go.Histogram( x=male_age, histnorm='probability', name="German Male", marker = dict( color = 'rgba(100, 149, 237, 0.6)', ) ) 
trace1 = go.Histogram( x=female_age, histnorm='probability', name="German Female", marker = dict( color = 'rgba(255, 182, 193, 0.6)', ) ) 
trace2 = go.Histogram( x=by_age, histnorm='probability', name="Overall Gender", marker = dict( color = 'rgba(169, 169, 169, 0.6)', ) ) 
fig = tools.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]], subplot_titles=('Males','Female', 'All Genders')) 

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)

fig['layout'].update(showlegend=True, title='Distribution of Gender', bargap=0.05)
iplot(fig, filename='custom-sized-subplot-with-subplot-titles')


# In[ ]:


# Gender vs Purpose let's see the purpose of having credit loans for each gender.
df["Purpose"].unique()
sex_purpose = pd.crosstab(df['Purpose'], df['Sex']).apply(lambda x: x/x.sum() * 100)
sex_purpose


# In[ ]:


# Values into list
m_purpose = sex_purpose['male'].values.tolist()
f_purpose = sex_purpose['female'].values.tolist()

# Round values
male_purpose = ['%.2f' % elem for elem in m_purpose]
female_purpose = ['%.2f' % elem for elem in f_purpose]

male = go.Bar( y=['business', 'car', 'domestic appliances', 'education', 'furniture/equipment', 'radio/TV', 'repairs', 'vacation/others'], x=male_purpose, name='German Males', text='(%)', orientation = 'h', marker = dict( color = 'rgba(72, 92, 238, 0.6)', line = dict( color = 'rgba(72, 92, 238, 1.0)', width = 3) ) ) 
female = go.Bar( y=['business', 'car', 'domestic appliances', 'education', 'furniture/equipment', 'radio/TV', 'repairs', 'vacation/others'], x=female_purpose, name='German Females', text='(%)', orientation = 'h', marker = dict( color = 'rgba(217, 128, 234, 0.6)', line = dict( color = 'rgba(217, 128, 234, 1.0)', width = 3) ) ) 

data = [male, female]
layout = dict( title='Application Reasons for a Loan by Gender', xaxis=dict( title='(% of Gender)', titlefont=dict( family='Arial, sans-serif', size=11, color='black' ), showticklabels=True, tickangle=45, tickfont=dict( family='Old Standard TT, serif', size=14, color='black' ), exponentformat='e', showexponent='all' ), yaxis=dict( title='Purpose for the Loan', titlefont=dict( family='Arial, sans-serif', size=11, color='black' ), showticklabels=True, tickangle=40, tickfont=dict( family='Old Standard TT, serif', size=10, color='black' ), exponentformat='e', showexponent='all' ), barmode='stack' ) 

fig = dict(data=data, layout=layout)
iplot(fig, filename='marker-h-bar')


# In[ ]:


# Percent distribution of Gender (how many jobs does each gender have in %)
number_of_jobs = pd.crosstab(df["Job"], df["Sex"]).apply(lambda x: x/x.sum() * 100)
number_of_jobs


# In[ ]:


# # Values to list
# job_number_m = number_of_jobs["male"].values.tolist()
# job_number_f = number_of_jobs["female"].values.tolist()

# # Round elements
# job_number_male = ["%.2f" % element for element in job_number_m]
# job_number_female = ["%.2f" % element for element in job_number_f]

# fig = {
#   "data": [
#     {
#       "values": job_number_male,
#       "labels": [
#         "Unemployed",
#         "One Job",
#         "Two Jobs",
#         "Three Jobs"
#       ],
#         'marker': {'colors': ['rgb(232, 71, 64)',
#                              'rgb(64, 232, 102)',
#                              'rgb(64, 181, 232)',
#                              'rgb(163, 64, 232)'],
#                   'line': {
#                   'color': '#000000', 'width':2,
#               },
#                   },
#         "text": "Males",
#       "domain": {"x": [0, .48]},
#       "name": "German Males",
#       "hoverinfo":"label+percent+name",
#       "hole": .4,
#       "type": "pie"
#     },     
#     {
#       "values": job_number_female,
#       "labels": [
#         "Unemployed",
#         "One Job",
#         "Two Jobs",
#         "Three Jobs"
#       ],
#     'marker': {'colors': ['rgb(232, 71, 64)',
#                              'rgb(64, 232, 102)',
#                              'rgb(64, 181, 232)',
#                              'rgb(163, 64, 232)'],
#               'line': {
#                   'color': '#000000', 'width':2,
#               },
#               },
#       "text":"Females",
#       "textposition":"inside",
#       "domain": {"x": [.52, 1]},
#       "name": "German Females",
#       "hoverinfo":"label+percent+name",
#       "hole": .4,
#       "type": "pie"
#     }],
#   "layout": {
#         "title":"Number of Jobs taken by Gender",
#       "font": {
#           "size": 14
#               },
#         "annotations": [
#             {
#                 "font": {
#                     "size": 12
#                 },
#                 "showarrow": False,
#                 "text": "Males",
#                 "x": 0.20,
#                 "y": 0.5
#             },
#             {
#                 "font": {
#                     "size": 12
#                 },
#                 "showarrow": False,
#                 "text": "Females",
#                 "x": 0.815,
#                 "y": 0.5
#             }
#         ]
#     }
# }
# iplot(fig, filename='donut')


# ## Age Groups:
# <a id="age_groups"></a>
# In this section we will create categorical groups based on the age column. The following categorical variables will belong to the <b> "Age_Group"</b> column:
# <ul>
# <li> <b>Young: </b> Clients  age ranges from (19 - 29). </li>
# <li> <b>Young Adults: </b> Clients age ranges from (30-40) </li>
# <li> <b> Senior: </b>  Clients age ranges from (41-55) </li>
# <li> <b>Elder: </b> Clients age is more than 55 years old </li>
# </ul>
# 
# ## What we want to accomplish:
# <ul>
# <li> Create different age groups based on their age. </li>
# <li> See the Credit amounts borrowed by clients belonging to each age group.</li>
# <li> Get deeper in our analysis and determine which loans were high risk and see if there is any patterns with regards to age groups.  </li>
# </ul>
# 
# ## Summary:
# <ul>
# <li> The <b>younger age group</b> tended to ask slightly for higher loans compared to the older age groups. </li>
# <li> The young and elederly groups had the <b>highest ratio</b> of high risk loans. With <b>45.29%</b> of all the clients that belong to the young age group being considered of high risk. </li>
# <li> The number of loans that were considered of high risk within the elderly group is<b> 44.28%</b> of the total amount of people considered in the elderly group. </li>
# <li> Interesting enough these are the groups that are most likely to be<b> unemployed or working part-time</b>, since the youngest group either <b>don't have the experience</b> to have a job or they are <b>studying in a university</b> so they don't have enough time to work in a full-time job. </li>
# <li>In the elderly group side, this is the group that are most likely receiving their money from their <b>pensions</b>, meaning the elderly group is most likely unemployed or working part-time. </li>
# </ul>

# In[ ]:


# Ok we have to create for each group risky and non-risky loans.
df['Age_Group'] = np.nan

lst = [df]

for col in lst:
    col.loc[(col['Age'] > 18) & (col['Age'] <= 29), 'Age_Group'] = 'Young'
    col.loc[(col['Age'] > 29) & (col['Age'] <= 40), 'Age_Group'] = 'Young Adults'
    col.loc[(col['Age'] > 40) & (col['Age'] <= 55), 'Age_Group'] = 'Senior'
    col.loc[col['Age'] > 55, 'Age_Group'] = 'Elder' 
    
df.head()


# In[ ]:


# Take credit amount values into a list
young = df['Credit_amount'].loc[df['Age_Group'] == 'Young'].values.tolist()
young_adults = df['Credit_amount'].loc[df['Age_Group'] == 'Young Adults'].values.tolist()
senior = df['Credit_amount'].loc[df['Age_Group'] == 'Senior'].values.tolist()
elder_credit = df['Credit_amount'].loc[df['Age_Group'] == 'Elder'].values.tolist()

# Create the box plots by age category
young_credit = go.Box( y = young, name = "Young", jitter = 0.3, pointpos = -1.8, boxpoints = 'all', marker = dict( color = 'rgb(150, 198, 109)'), line = dict( color = 'rgb(111, 200, 37)') ) 

young_adults_credit = go.Box( y = young_adults, name = "Young Adults", jitter = 0.3, pointpos = -1.8, boxpoints = 'all', marker = dict( color = 'rgb(124, 236, 212)'), line = dict( color = 'rgb(38, 214, 177)') ) 

senior_credit = go.Box( y = senior, name = "Seniors", jitter = 0.3, pointpos = -1.8, boxpoints = 'all', marker = dict( color = 'rgb(241, 93, 93)'), line = dict( color = 'rgb(225, 44, 44)') ) 

elder_credit = go.Box( y = elder_credit, name = "Elders", jitter = 0.3, pointpos = -1.8, boxpoints = 'all', marker = dict( color = 'rgb(180, 121, 72)'), line = dict( color = 'rgb(115, 77, 46)') ) 

data = [young_credit, young_adults_credit, senior_credit, elder_credit]

layout = dict( title="Credit Amount by Age Group Segment", xaxis = dict(title="Age Group"), yaxis= dict(title="Credit Amount") ) 

fig = dict(data=data, layout=layout)
iplot(fig, filename="Box Plot")


# In[ ]:


good_credit_df = df.loc[df['Risk'] == "good"]
bad_credit_df = df.loc[df['Risk'] == "bad"]

good_credit_df.head()


# In[ ]:


fig = { "data": [ { "type": 'violin', "x": good_credit_df['Age_Group'], "y": good_credit_df['Credit_amount'], "legendgroup": 'Good Credit', "scalegroup": 'No', "name": 'Good Credit', "side": 'negative', "box": { "visible": True }, "meanline": { "visible": True }, "line": { "color": 'rgb(34, 178, 178)' } }, { "type": 'violin', "x": bad_credit_df['Age_Group'], "y": bad_credit_df['Credit_amount'], "legendgroup": 'Bad Credit', "scalegroup": 'No', "name": 'Bad Credit', "side": 'positive', "box": { "visible": True }, "meanline": { "visible": True }, "line": { "color": 'rgb(178, 34, 34)' } } ], "layout" : { "title": "Distribution of Credit Borrowed by Age Group", "yaxis": { "zeroline": False, }, "violingap": 0, "violinmode": "overlay" } } 


iplot(fig, filename = 'Age-Housing', validate=False)


# In[ ]:


# Lets find loans by age group and by the level of risk and plot them in a bar chart.

# Age Group Segments
young_good = df['Credit_amount'].loc[(df['Age_Group'] == 'Young') & (df['Risk'] == 'good')].sum()
young_bad = df['Credit_amount'].loc[(df['Age_Group'] == 'Young') & (df['Risk'] == 'bad')].sum()
young_adult_good = df['Credit_amount'].loc[(df['Age_Group'] == 'Young Adults') & (df['Risk'] == 'good')].sum()
young_adult_bad = df['Credit_amount'].loc[(df['Age_Group'] == 'Young Adults') & (df['Risk'] == 'bad')].sum()
senior_good = df['Credit_amount'].loc[(df['Age_Group'] == 'Senior') & (df['Risk'] == 'good')].sum()
senior_bad = df['Credit_amount'].loc[(df['Age_Group'] == 'Senior') & (df['Risk'] == 'bad')].sum()
elder_good = df['Credit_amount'].loc[(df['Age_Group'] == 'Elder') & (df['Risk'] == 'good')].sum()
elder_bad = df['Credit_amount'].loc[(df['Age_Group'] == 'Elder') & (df['Risk'] == 'bad')].sum()

# Percents
young_good_p = young_good/(young_good + young_bad) * 100
young_bad_p = young_bad/(young_good + young_bad) * 100
young_adult_good_p = young_adult_good/(young_adult_good + young_adult_bad) * 100
young_adult_bad_p = young_adult_bad/(young_adult_good + young_adult_bad) * 100
senior_good_p = senior_good/(senior_good + senior_bad) * 100
senior_bad_p =  senior_bad/(senior_good + senior_bad) * 100
elder_good_p = elder_good/(elder_good + elder_bad) * 100
elder_bad_p = elder_bad/(elder_good + elder_bad) * 100

# Round Percents
young_good_p = str(round(young_good_p, 2))
young_bad_p = str(round(young_bad_p, 2))
young_adult_good_p = str(round(young_adult_good_p, 2))
young_adult_bad_p = str(round(young_adult_bad_p, 2))
senior_good_p = str(round(senior_good_p, 2))
senior_bad_p = str(round(senior_bad_p, 2))
elder_good_p = str(round(elder_good_p, 2))
elder_bad_p = str(round(elder_bad_p, 2))



x = ["Young", "Young Adults", "Senior", "Elder"]

good_loans = go.Bar( x=x, y=[young_good, young_adult_good, senior_good, elder_good], name="Good Loans", text=[young_good_p + '%', young_adult_good_p + '%', senior_good_p + '%', elder_good_p + '%'], textposition = 'auto', marker=dict( color='rgb(111, 235, 146)', line=dict( color='rgb(60, 199, 100)', width=1.5), ), opacity=0.6 ) 

bad_loans =  go.Bar( x=x, y=[young_bad, young_adult_bad, senior_bad, elder_bad], name="Bad Loans", text=[young_bad_p + '%', young_adult_bad_p + '%', senior_bad_p + '%', elder_bad_p + '%'], textposition = 'auto', marker=dict( color='rgb(247, 98, 98)', line=dict( color='rgb(225, 56, 56)', width=1.5), ), opacity=0.6 ) 

data = [good_loans, bad_loans]

layout = dict( title="Type of Loan by Age Group", xaxis = dict(title="Age Group"), yaxis= dict(title="Credit Amount") ) 

fig = dict(data=data, layout=layout)

iplot(fig, filename='grouped-bar-direct-labels')


# ## Wealth Analysis:
# <a id="wealth_analysis"></a>
# In this section we will analyse the amount of wealth our clients have by analyzing their checking accounts and whether the wealth status of our clients contribute to the risk of the loans Lending Club is issuing to customers.
# 
# ## Summary: 
# <ul>
# <li> Individuals belonging to the <b>"little wealth"</b>  group, had a higher probability of being bad risk loans than other types fo groups.</li>
# <li> The <b>higher the wealth</b>, the lower the probability of being a bad risk loan. </li>
# </ul>

# In[ ]:


# We have some missing value so we will just ignore the missing values in this analysis.
df["Checking account"].unique()
df.columns


# In[ ]:


cross_checking = pd.crosstab(df['Risk'], df['Checking account']).apply(lambda x: x/x.sum() * 100)
decimals = pd.Series([2,2,2], index=['little', 'moderate', 'rich'])

cross_checking = cross_checking.round(decimals)
cross_checking


# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go


bad_little = cross_checking['little'][0]
good_little = cross_checking['little'][1]
bad_moderate = cross_checking['moderate'][0]
good_moderate = cross_checking['moderate'][1]
bad_rich = cross_checking['rich'][0]
good_rich = cross_checking['rich'][1]

little = go.Bar( x=['Bad Risk', 'Good Risk'], y=[bad_little, good_little], name="Little Wealth", text='%', marker=dict(         color='#FF5050', line=dict(             color='#E13232', width=1.5), ), ) 

moderate = go.Bar( x=['Bad Risk', 'Good Risk'], y=[bad_moderate, good_moderate], name="Moderate Wealth", text='%', xaxis='x2', yaxis='y2', marker=dict(         color='#FFB232', line=dict(             color='#CD8000', width=1.5), ), ) 
rich = go.Bar( x=['Bad Risk', 'Good Risk'], y=[bad_rich, good_rich], name="Rich Wealth", text="%", marker=dict(         color='#8DFF83', line=dict(             color='#3DAF33', width=1.5), ), xaxis='x3', yaxis='y3' ) 

data=[little, moderate, rich]

fig = tools.make_subplots(rows=1, cols=3, subplot_titles=('Little Wealth', 'Moderate Wealth', 'Rich Wealth'))

fig.append_trace(little, 1, 1)
fig.append_trace(moderate, 1, 2)
fig.append_trace(rich, 1, 3)

fig['layout'].update(height=400, width=800, title='Levels of Risk' + ' by Wealth') 


iplot(fig, filename='make-subplots-multiple-with-titles')


# ## High Risk Loans vs Low Risk Loans:
# In this section we will analyze both high and low risk loans. The most important thing is to find patters that could describe the some sort of correlation with these output values.
# 
# ## Correlation (Our intent):
# <a id="correlations"></a>
# In this part of the analysis, we want to look as to what feature affect directly the risk of the loan. In order to see these patterns, the first thing we have to do is to create a new column named "Risk_int" (Stands for risk in integer form) and involve this column in the correlation heatmap plot. "0" will stand for "bad risk" loans and "1" will stand for "good risk" loans.
# 
# ## Summary:
# <ul>
# <li> The higher the <b>credit amount</b> borrowed, the most likely the loan will end up <b>bad</b>.</li>
# <li> The higher the <b>duration</b> of the loan, the most likely the loan will turn out to be <b> bad</b></li>
# <li><b>Senior</b> and <b> Elders </b> that asked for loans over 12k, have a high chance of becoming <b>bad loans </b> </li>
# <li>If the credit amount borrowed is <b>equivalent to 11,000 or more</b>, the probability for the loan to be a bad one increases drastically. (Observe the Correlation of Risk with Credit Amount Borrowed.) </li>
# 
# </ul>

# In[ ]:


df['Risk_int'] = np.nan
lst = [df]

for col in lst:
    col.loc[df['Risk'] == 'bad', 'Risk_int'] = 0 
    col.loc[df['Risk'] == 'good', 'Risk_int'] = 1
    
    
df['Risk_int'] = df['Risk_int'].astype(int)
df.head()


# In[ ]:


import plotly.plotly as py
import plotly.figure_factory as ff


corr = df.corr()

arr_corr = corr.values
arr_corr = np.around(arr_corr, decimals=2)

columns = corr.columns.values.tolist()

fig = ff.create_annotated_heatmap(arr_corr, x=columns, y=columns, colorscale='Reds')

fig.layout.title = 'Pearson Correlation <br> (What Determines Risk in Loans?)'
iplot(fig, filename='annotated_heatmap')


# In[ ]:


# ---> Looking at correlations in the form of Scatterplots.

import plotly.plotly as py
import plotly.figure_factory as ff

import numpy as np
import pandas as pd

numeric_vals = df.select_dtypes(exclude=['object'])


risks = df['Risk'].values.tolist()

colormap = ['#b22222', '#4169e1']

numeric_matrix = numeric_vals.as_matrix()


dataframe = pd.DataFrame(numeric_matrix, columns=['Credit_amount', 'Duration', 'Job', 'Age', 'Risk']) 

dataframe['Risk'] = pd.Series(risks)


fig = ff.create_scatterplotmatrix(dataframe, diag='histogram', index='Risk', colormap=colormap, height=800, width=800) 
iplot(fig, filename='Histograms along Diagonal Subplots')


# In[ ]:


# The higher the credit amount the higher the risk of the loan. Scatter plot?
# The higher the duration of the loan the higher the risk of the loan?

bad_credit_amount = df["Credit_amount"].loc[df['Risk'] == 'bad'].values.tolist()
good_credit_amount = df["Credit_amount"].loc[df['Risk'] == 'good'].values.tolist()
bad_duration = df['Duration'].loc[df['Risk'] == 'bad'].values.tolist()
good_duration = df['Duration'].loc[df['Risk'] == 'good'].values.tolist()


bad_loans = go.Scatter( x = bad_duration, y = bad_credit_amount, name = 'Bad Loans', mode = 'markers', marker = dict( size = 10, color = 'rgba(152, 0, 0, .8)', line = dict( width = 2, color = 'rgb(0, 0, 0)' ) ) ) 

good_loans = go.Scatter( x = good_duration, y = good_credit_amount, name = 'Good Loans', mode = 'markers', marker = dict( size = 10, color = 'rgba(34, 139, 34, .9)', line = dict( width = 2, ) ) ) 

data = [bad_loans, good_loans]

layout = dict(title = 'Correlation of Risk with <br> Credit Amount Borrowed', yaxis = dict(zeroline = False), xaxis = dict(zeroline = False) ) 

fig = dict(data=data, layout=layout)
iplot(fig, filename='styled-scatter')


# ## Exploring Purposes of Loans:
# <a id="purpose_loans"></a>
# In this section my main aim is to see what purposes where most likely to bring most risk, in other words which of these pruposes were more likely to be considered high risk loans. Also, I would like to explore the operative side of the business, by determining which purposes where the ones that contributed the most towards loans issued.
# 
# ## Summary: 
# <ul> 
# <li><b>Cars</b>, <b> Radio/TV</b> and <b> Furniture and Equipment</b> made more than 50 % of the total risk and has the <b>highest distribution</b> of credit issued</li>
# <li>The <b> rest of the purposes</b> were not frequent purposes in applying for a loan. </li>
# <li> <b>Cars</b> and <b>Radio/TV</b> purposes were the less risky from the operative perspective since it had the widest gap between good and bad risk. </li>
# </ul>

# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go


radio_tv = np.sum(df['Credit_amount'].loc[df['Purpose'] == 'radio/TV'].values)
education = np.sum(df['Credit_amount'].loc[df['Purpose'] == 'education'].values)
furniture = np.sum(df['Credit_amount'].loc[df['Purpose'] == 'furniture/equipment'].values)
car = np.sum(df['Credit_amount'].loc[df['Purpose'] == 'car'].values)
business = np.sum(df['Credit_amount'].loc[df['Purpose'] == 'business'].values)
domestic_app = np.sum(df['Credit_amount'].loc[df['Purpose'] == 'domestic appliances'].values)
repairs = np.sum(df['Credit_amount'].loc[df['Purpose'] == 'repairs'].values)
vacation_others = np.sum(df['Credit_amount'].loc[df['Purpose'] == 'vacation/others'].values)

df_purposes = pd.DataFrame(data=[[radio_tv, education, furniture, car, business, domestic_app, repairs, vacation_others]], columns=df["Purpose"].unique()) 

print(df_purposes)

data = [ go.Scatterpolar( r = [radio_tv, education, furniture,car, business, domestic_app, repairs, vacation_others], theta = df['Purpose'].unique().tolist(), line = dict(             color = "#ff7f50" ), mode = 'lines+markers', fill='toself', marker = dict(             color = '#8090c7', symbol='square', size=6 ) ) ] 

layout = go.Layout( showlegend = False, title= "Credit Distribution by Purpose", paper_bgcolor = "rgb(255, 245, 250)" ) 

fig = dict(data=data, layout=layout)
iplot(fig, filename = 'polar/basic')


# In[ ]:


df['Purpose'].unique()

cross_purpose = pd.crosstab(df['Purpose'], df['Risk']).apply(lambda x: x/x.sum() * 100)
cross_purpose = cross_purpose.round(decimals=2)
cross_purpose.sort_values(by=['bad'])


# In[ ]:


# This will be x
purposes = df['Purpose'].unique().tolist()
purposes = sorted(purposes)

good_purposes = cross_purpose["good"].values.tolist()
bad_purposes = cross_purpose["bad"].values.tolist()




good = go.Bar( x=purposes, y=good_purposes, name='Good Risk', text='%', marker=dict( color='rgb(123, 255, 255)', line=dict( color='rgb(0, 213, 213)', width=1.5 ) ), ) 
bad = go.Bar( x=purposes, y=bad_purposes, name='Bad Risk', text = '%', marker=dict( color='rgb(255, 123, 123)', line=dict( color='rgb(213, 0, 0)', width=1.5 ) ), ) 
data = [good, bad]
layout = go.Layout( title='Which Purpose Carried the Highest Risk?', xaxis=dict( tickfont=dict( size=14, color='rgb(107, 107, 107)' ) ), yaxis=dict( title='Percentage(%)', titlefont=dict( size=16, color='rgb(107, 107, 107)' ), tickfont=dict( size=14, color='rgb(107, 107, 107)' ) ), legend=dict( x=0, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)' ), barmode='group', bargap=0.15, bargroupgap=0.1 ) 

fig = dict(data=data, layout=layout)
iplot(fig, filename='style-bar')


# In[ ]:


cross_credit = df.groupby(['Risk', 'Purpose'], as_index=False).Credit_amount.sum()

good_amnt = cross_credit["Credit_amount"].loc[cross_credit["Risk"] == "good"].values.tolist()
bad_amnt = cross_credit["Credit_amount"].loc[cross_credit["Risk"] == "bad"].values.tolist()

good_risk = {"x": good_amnt, "y": purposes, "marker": {"color": "rgb(28, 98, 98)", "size": 12}, "mode": "markers", "name": "Good Risk", "type": "scatter" } 

bad_risk = {"x": bad_amnt, "y": purposes, "marker": {"color": "rgb(98, 28, 28)", "size": 12}, "mode": "markers", "name": "Bad Risk", "type": "scatter", } 

data = [good_risk, bad_risk]
layout = {"title": "Credit Loan Applications by Purpose", "xaxis": {"title": "Credit Amount Borrowed", }, "yaxis": {"title": ""}, "width": 800, "height": 700} 

fig = dict(data=data, layout=layout)
iplot(fig, filename='basic_dot-plot')


# ## Predictive Modelling:
# <a id="predictive_modelling"></a>
# 

# In[ ]:


# Check missing values in our dataframe
original_df.isnull().sum().sort_values(ascending=False)


# In[ ]:


# We will drop the columns that have missing values although we will be loosing some information. Hopefully this does not cause
# the model to underfit in the future.
original_df.drop(['Checking account', 'Saving accounts'], axis=1, inplace=True)


# In[ ]:


original_df.isnull().sum().sort_values(ascending=False)


# In[ ]:


# Feature Engineering (We cannot delete the missing values because we have too litle information)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit

original_df["Risk"].value_counts() # 70% is good risk and 30% is bad risk.

stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train, test in stratified.split(original_df, original_df["Risk"]):
    strat_train = original_df.loc[train]
    strat_test = original_df.loc[test]
    

# The main purpose of this code is to have an approximate ratio
# of 70% good risk and 30% bad risk in both training and testing sets.
strat_train["Risk"].value_counts() / len(df) 
strat_test["Risk"].value_counts() / len(df)


# In[ ]:


# Have our new train and test data
train = strat_train
test = strat_test


# Our features
X_train = train.drop('Risk', axis=1)
X_test = test.drop('Risk', axis=1)

# Our Labels we will use them later
y_train = train["Risk"]
y_test = test["Risk"]


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array. The input to this transformer should be a matrix of integers or strings, denoting the values taken on by categorical (discrete) features. The features can be encoded using a one-hot aka one-of-K scheme (``encoding='onehot'``, the default) or converted to ordinal integers (``encoding='ordinal'``). This encoding is needed for feeding categorical data to many scikit-learn estimators, notably linear models and SVMs with the standard kernels. Read more in the :ref:`User Guide <preprocessing_categorical_features>`. Parameters ---------- encoding : str, 'onehot', 'onehot-dense' or 'ordinal' The type of encoding to use (default is 'onehot'): - 'onehot': encode the features using a one-hot aka one-of-K scheme (or also called 'dummy' encoding). This creates a binary column for each category and returns a sparse matrix. - 'onehot-dense': the same as 'onehot' but returns a dense array instead of a sparse matrix. - 'ordinal': encode the features as ordinal integers. This results in a single column of integers (0 to n_categories - 1) per feature. categories : 'auto' or a list of lists/arrays of values. Categories (unique values) per feature: - 'auto' : Determine categories automatically from the training data. - list : ``categories[i]`` holds the categories expected in the ith column. The passed categories are sorted before encoding the data (used categories can be found in the ``categories_`` attribute). dtype : number type, default np.float64 Desired dtype of output. handle_unknown : 'error' (default) or 'ignore' Whether to raise an error or ignore if a unknown categorical feature is present during transform (default is to raise). When this is parameter is set to 'ignore' and an unknown category is encountered during transform, the resulting one-hot encoded columns for this feature will be all zeros. Ignoring unknown categories is not supported for ``encoding='ordinal'``. Attributes ---------- categories_ : list of arrays The categories of each feature determined during fitting. When categories were specified manually, this holds the sorted categories (in order corresponding with output of `transform`). Examples -------- Given a dataset with three features and two samples, we let the encoder find the maximum value per feature and transform the data to a binary one-hot encoding. >>> from sklearn.preprocessing import CategoricalEncoder >>> enc = CategoricalEncoder(handle_unknown='ignore') >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]) ...  CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>, encoding='onehot', handle_unknown='ignore') >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray() array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.], [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]]) See also -------- sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of integer ordinal features. The ``OneHotEncoder assumes`` that input features take on values in the range ``[0, max(feature)]`` instead of using the unique values. sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of dictionary items (also handles string-valued features). sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot encoding of dictionary items or strings. """ 

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64, handle_unknown='error'): 
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X. Parameters ---------- X : array-like, shape [n_samples, n_feature] The data to determine the categories of each feature. Returns ------- self """ 

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' " "or 'ordinal', got %s") 
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or " "'ignore', got %s") 
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for" " encoding='ordinal'") 

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}" " during fit".format(diff, i)) 
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding. Parameters ---------- X : array-like, shape [n_samples, n_features] The data to encode. Returns ------- X_out : sparse matrix or a 2-d array Transformed input. """ 
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}" " during transform".format(diff, i)) 
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32), n_features)[mask] 
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)), shape=(n_samples, indices[-1]), dtype=self.dtype).tocsr() 
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out


# In[ ]:


# Scikit-Learn does not handle dataframes in pipeline so we will create our own class.
# Reference: Hands-On Machine Learning
from sklearn.base import BaseEstimator, TransformerMixin
# Create a class to select numerical or cateogrical columns.
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit (self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[ ]:


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

numeric_train_df = X_train.select_dtypes(exclude=['object'])
numeric_test_df = X_test.select_dtypes(exclude=['object'])

categorical_train_df = X_train.select_dtypes(['object'])
categorical_test_df = X_test.select_dtypes(['object'])

numerical_pipeline = Pipeline([ ("select_numeric", DataFrameSelector(numeric_train_df.columns.values.tolist())), ("std_scaler", StandardScaler()), ]) 

categorical_pipeline = Pipeline([ ('select_categoric', DataFrameSelector(categorical_train_df.columns.values.tolist())), ('encoding', CategoricalEncoder(encoding='onehot-dense')) ]) 

# Combine both pipelines
main_pipeline = FeatureUnion(transformer_list=[ ('num_pipeline', numerical_pipeline), ('cat_pipeline', categorical_pipeline) ]) 

X_train_scaled = main_pipeline.fit_transform(X_train)
X_test_scaled = main_pipeline.fit_transform(X_test)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

encode = LabelEncoder()
y_train_scaled = encode.fit_transform(y_train)
y_test_scaled = encode.fit_transform(y_test)


# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Implement gridsearchcv to see which are our best p

params = {'C': [0.75, 0.85, 0.95, 1], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [3, 4, 5]}

svc_clf = SVC(random_state=42)

grid_search_cv = GridSearchCV(svc_clf, params)
grid_search_cv.fit(X_train_scaled, y_train_scaled)


# In[ ]:


grid_search_cv.best_estimator_


# In[ ]:


grid_search_cv.best_params_


# In[ ]:


svc_clf = grid_search_cv.best_estimator_
svc_clf.fit(X_train_scaled, y_train_scaled)


# In[ ]:


svc_clf.score(X_train_scaled, y_train_scaled)


# In[ ]:


from sklearn.model_selection import cross_val_score

# Let's make sure the data is not overfitting
svc_clf = SVC(kernel='rbf', C=1, random_state=42)
scores = cross_val_score(svc_clf, X_train_scaled, y_train_scaled)
scores.mean()


# In[ ]:


from sklearn.metrics import accuracy_score

svc_clf.fit(X_train_scaled, y_train_scaled)
y_pred = svc_clf.predict(X_test_scaled)

# Accuracy score 71%
accuracy_score(y_test_scaled, y_pred)


# ## References:
# <ul> 
# <li> <a href="https://www.kaggle.com/kabure/german-credit-risk-financial-eda"> German Credit risk [FinancialEDA]</a> by Leonardo Ferreira</li>
# 
# </ul>
# 
# ### Note: This Kernel will be subjected to further updates.
