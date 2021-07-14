#!/usr/bin/env python
# coding: utf-8

# ## Getting Started
# In this project, we will evaluate the performance and predictive power of a model that has been trained and tested on data collected from homes in suburbs of Boston, Massachusetts. A model trained on this data that is seen as a *good fit* could then be used to make certain predictions about a home — in particular, its monetary value. This model would prove to be invaluable for someone like a real estate agent who could make use of such information on a daily basis.
# 
# The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing). The Boston housing data was collected in 1978 and each of the 506 entries represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts. For the purposes of this project, the following preprocessing steps have been made to the dataset:
# - 16 data points have an `'MEDV'` value of 50.0. These data points likely contain **missing or censored values** and have been removed.
# - 1 data point has an `'RM'` value of 8.78. This data point can be considered an **outlier** and has been removed.
# - The features `'RM'`, `'LSTAT'`, `'PTRATIO'`, and `'MEDV'` are essential. The remaining **non-relevant features** have been excluded.
# - The feature `'MEDV'` has been **multiplicatively scaled** to account for 35 years of market inflation.
# 
# Run the code cell below to load the Boston housing dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_validation import ShuffleSplit

# Pretty display for notebooks
print()
# Input data files are available in the "../../../input/schirmerchad_bostonhoustingmlnd/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../../../input/schirmerchad_bostonhoustingmlnd"]).decode("utf8"))



# In[ ]:


# Load the Boston housing dataset
data = pd.read_csv("../../../input/schirmerchad_bostonhoustingmlnd/housing.csv")
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

data.head()


# ## Data Exploration
# In this first section of this project, we will make a cursory investigation about the Boston housing data and provide our observations. Familiarizing ourself with the data through an explorative process is a fundamental practice to help us better understand and justify our results.
# 
# Since the main goal of this project is to construct a working model which has the capability of predicting the value of houses, we will need to separate the dataset into **features** and the **target variable**. The **features**, `'RM'`, `'LSTAT'`, and `'PTRATIO'`, give us quantitative information about each data point. The **target variable**, `'MEDV'`, will be the variable we seek to predict. These are stored in `features` and `prices`, respectively.

# ### Implementation: Calculate Statistics
# For our very first coding implementation, we will calculate descriptive statistics about the Boston housing prices. Since `numpy` has already been imported for us, use this library to perform the necessary calculations. These statistics will be extremely important later on to analyze various prediction results from the constructed model.
# 
# In the code cell below, we will need to implement the following:
# - Calculate the minimum, maximum, mean, median, and standard deviation of `'MEDV'`, which is stored in `prices`.
#   - Store each calculation in their respective variable.

# In[ ]:


# TODO: Minimum price of the data
minimum_price = np.mean(prices)

# TODO: Maximum price of the data
maximum_price = np.max(prices)

# TODO: Mean price of the data
mean_price = np.mean(prices)

# TODO: Median price of the data
median_price = np.median(prices)

# TODO: Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price))
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))


# ### Question 1 - Feature Observation
# As a reminder, we are using three features from the Boston housing dataset: `'RM'`, `'LSTAT'`, and `'PTRATIO'`. For each data point (neighborhood):
# - `'RM'` is the average number of rooms among homes in the neighborhood.
# - `'LSTAT'` is the percentage of homeowners in the neighborhood considered "lower class" (working poor).
# - `'PTRATIO'` is the ratio of students to teachers in primary and secondary schools in the neighborhood.
# 
# 
# ** Using your intuition, for each of the three features above, do you think that an increase in the value of that feature would lead to an **increase** in the value of `'MEDV'` or a **decrease** in the value of `'MEDV'`? Justify your answer for each.**
# 
# **Hint:** This problem can phrased using examples like below.  
# * Would you expect a home that has an `'RM'` value(number of rooms) of 6 be worth more or less than a home that has an `'RM'` value of 7?
# * Would you expect a neighborhood that has an `'LSTAT'` value(percent of lower class workers) of 15 have home prices be worth more or less than a neighborhood that has an `'LSTAT'` value of 20?
# * Would you expect a neighborhood that has an `'PTRATIO'` value(ratio of students to teachers) of 10 have home prices be worth more or less than a neighborhood that has an `'PTRATIO'` value of 15?

# **Answer: ** In my opinion, the value of 'MEDV' will be dependent on these 3 features in the following way:
# 
# 1) **RM** - The more the value of RM, the more will be the value of 'MEDV'. Because it's pretty evident that with increase in the number of rooms, the price of the house will increase.
# 
# 2) **LSTAT** - The more the value of LSTAT, the less will be the value of 'MEDV'. Because with increase in the percentage of "lower class" homeowners in the neighbourhood, the crime rate in the neighbourhood may increase. Even though LSTAT doesn't have a causal effect on the crime rate in the neighbourhood, they are likely to be positively correlated. One more factor is if there are greater percentages of "lower class" homeowners in the neighbourhood, then more likely very expensive real estate owners will not build their housing complexes in that region as most of the people will not be able to afford it. So in average, the houses in that region will be cheaper.
# 
# 3) **PTRATIO** - The lesser the value of PTRATIO, the more will be the value of 'MEDV'. Because if the students to teacher ratio is low, then that means individual students gets much more attention from the students as opposed to a region where this ratio is high. Over there, as the number of students will be much higher than the number of teachers, teachers will not be able to attend to students individually everytime and hence this may affect the education of the students. So regions with a low PTRATIO will have higher prices for houses.

# ## Initial Visualization

# In[ ]:


# Using pyplot
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5))

# i: index
for i, col in enumerate(features.columns):
    # 3 plots here hence 1, 3
    plt.subplot(1, 3, i+1)
    x = data[col]
    y = prices
    plt.plot(x, y, 'o')
    # Create regression line
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('prices')


# ----
# 
# ## Developing a Model
# In this second section of the project, we will develop the tools and techniques necessary for a model to make a prediction. Being able to make accurate evaluations of each model's performance through the use of these tools and techniques helps to greatly reinforce the confidence in our predictions.

# ### Implementation: Define a Performance Metric
# It is difficult to measure the quality of a given model without quantifying its performance over training and testing. This is typically done using some type of performance metric, whether it is through calculating some type of error, the goodness of fit, or some other useful measurement. For this project, we will be calculating the [*coefficient of determination*](http://stattrek.com/statistics/dictionary.aspx?definition=coefficient_of_determination), R<sup>2</sup>, to quantify our model's performance. The coefficient of determination for a model is a useful statistic in regression analysis, as it often describes how "good" that model is at making predictions. 
# 
# The values for R<sup>2</sup> range from 0 to 1, which captures the percentage of squared correlation between the predicted and actual values of the **target variable**. A model with an R<sup>2</sup> of 0 is no better than a model that always predicts the *mean* of the target variable, whereas a model with an R<sup>2</sup> of 1 perfectly predicts the target variable. Any value between 0 and 1 indicates what percentage of the target variable, using this model, can be explained by the **features**. _A model can be given a negative R<sup>2</sup> as well, which indicates that the model is **arbitrarily worse** than one that always predicts the mean of the target variable._
# 
# For the `performance_metric` function in the code cell below, we will need to implement the following:
# - Use `r2_score` from `sklearn.metrics` to perform a performance calculation between `y_true` and `y_predict`.
# - Assign the performance score to the `score` variable.

# In[ ]:


# TODO: Import 'r2_score'

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between true and predicted values based on the metric chosen. """ 
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    from sklearn.metrics import r2_score
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score


# ### Question 2 - Goodness of Fit
# Assume that a dataset contains five data points and a model made the following predictions for the target variable:
# 
# | True Value | Prediction |
# | :----------: | :--------: |
# | 3.0 | 2.5 |
# | -0.5 | 0.0 |
# | 2.0 | 2.1 |
# | 7.0 | 7.8 |
# | 4.2 | 5.3 |
# 
# Run the code cell below to use the `performance_metric` function and calculate this model's coefficient of determination.

# In[ ]:


# Calculate the performance of this model
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))


# ### Visualization

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
print()

true, pred = [3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3]

#Plot true values
true_handle = plt.scatter(true, true, alpha=0.6, color='blue', label='true')

#Reference line
fit = np.poly1d(np.polyfit(true,true,1))
lims = np.linspace(min(true) - 1, max(true) + 1)
plt.plot(lims, fit(lims), alpha=0.3, color='black')

#Plot predicted values
pred_handle = plt.scatter(true, pred, alpha=0.6, color='red', label='predicted')

#Legend and show
plt.legend(handles=[true_handle,pred_handle], loc='upper left')
print()


# * Would you consider this model to have successfully captured the variation of the target variable? 
# * Why or why not?
# 
# ** Hint: **  The R2 score is the proportion of the variance in the dependent variable that is predictable from the independent variable. In other words:
# * R2 score of 0 means that the dependent variable cannot be predicted from the independent variable.
# * R2 score of 1 means the dependent variable can be predicted from the independent variable.
# * R2 score between 0 and 1 indicates the extent to which the dependent variable is predictable. An 
# * R2 score of 0.40 means that 40 percent of the variance in Y is predictable from X.

# **Answer:** Yes, this model has successfully captured the variation of the target variable. This is because we are getting a very high R2 value of 0.923. That means 92.3% of the variance in the True Value is predictable from the Prediction. As this is a very high percentage, we can call this model to be a successful model.
# 
# The only drawback is there are only 5 datapoints here. So this might not be statistically significant. Another caveat is that whether the model is successful also depends largely on the application. So for some projects 0.923 is sufficient, whereas for others it could be a low score.

# ### Implementation: Shuffle and Split Data
# Our next implementation requires that we take the Boston housing dataset and split the data into training and testing subsets. Typically, the data is also shuffled into a random order when creating the training and testing subsets to remove any bias in the ordering of the dataset.
# 
# For the code cell below, we will need to implement the following:
# - Use `train_test_split` from `sklearn.cross_validation` to shuffle and split the `features` and `prices` data into training and testing sets.
#   - Split the data into 80% training and 20% testing.
#   - Set the `random_state` for `train_test_split` to a value of your choice. This ensures results are consistent.
# - Assign the train and testing splits to `X_train`, `X_test`, `y_train`, and `y_test`.

# In[ ]:


# TODO: Import 'train_test_split'
from sklearn import cross_validation

# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, prices, test_size = 0.2, random_state = 42)

# Success
print("Training and testing split was successful.")


# ### Question 3 - Training and Testing
# 
# * What is the benefit to splitting a dataset into some ratio of training and testing subsets for a learning algorithm?
# 
# **Hint:** Think about how overfitting or underfitting is contingent upon how splits on data is done.

# **Answer: ** A possible alternative to splitting a dataset into training and testing data would be to train and test on the same data. But that creates a problem. Here there is a very high chance of getting a high variance model which may eventually lead to a 100% accuracy rate with addition of new features, but that's only because it is overfitting the data. It has developed such a complex model that it will have limited or no ability to generalize data and so when we use that model on unknown data, it will give us very very low accuracy. So to avoid that, we can split the data into training and testing sets and train the model on the training data. Then the testing accuracy is a much better estimate than the training accuracy. 
# 
# But then, the split might create a problem too. If we have a very limited dataset, then even if we take out a small sample of it as testing data, then also , we are losing a portion of the data. So there's an inherent trade off here which might cause underfitting due to limited datasets. This is where we can take advantage of K-fold cross validation where we divide all the datapoints into k number of bins and then run k separate learning experiments. In each of those, we pick one of those k subsets as our testing set and the remaining k-1 bins as our training sets. This is how we can maximize the machine's learning experiment.

# ----
# 
# ## Analyzing Model Performance
# In this third section of the project, we'll take a look at several models' learning and testing performances on various subsets of training data. Additionally, we'll investigate one particular algorithm with an increasing `'max_depth'` parameter on the full training set to observe how model complexity affects performance. Graphing our model's performance based on varying criteria can be beneficial in the analysis process, such as visualizing behavior that may not have been apparent from the results alone.

# ### Learning Curves
# The following code cell produces four graphs for a decision tree model with different maximum depths. Each graph visualizes the learning curves of the model for both training and testing as the size of the training set is increased. Note that the shaded region of a learning curve denotes the uncertainty of that curve (measured as the standard deviation). The model is scored on both the training and testing sets using R<sup>2</sup>, the coefficient of determination.  
# 
# Run the code cell below and use these graphs to answer the following question.

# In[ ]:


#Define the necessary functions for plotting
###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
print()
###########################################

import matplotlib.pyplot as pl
import numpy as np
import sklearn.learning_curve as curves
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import ShuffleSplit, train_test_split

def ModelLearning(X, y):
    """ Calculates the performance of several models with varying sizes of training data. The learning and testing scores for each model are then plotted. """ 
    
    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)

    # Generate the training set sizes increasing by 50
    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)

    # Create the figure window
    fig = pl.figure(figsize=(10,7))

    # Create three different models based on max_depth
    for k, depth in enumerate([1,3,6,10]):
        
        # Create a Decision tree regressor at max_depth = depth
        regressor = DecisionTreeRegressor(max_depth = depth)

        # Calculate the training and testing scores
        sizes, train_scores, test_scores = curves.learning_curve(regressor, X, y,             cv = cv, train_sizes = train_sizes, scoring = 'r2')
        
        # Find the mean and standard deviation for smoothing
        train_std = np.std(train_scores, axis = 1)
        train_mean = np.mean(train_scores, axis = 1)
        test_std = np.std(test_scores, axis = 1)
        test_mean = np.mean(test_scores, axis = 1)

        # Subplot the learning curve 
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
        ax.fill_between(sizes, train_mean - train_std,             train_mean + train_std, alpha = 0.15, color = 'r')
        ax.fill_between(sizes, test_mean - test_std,             test_mean + test_std, alpha = 0.15, color = 'g')
        
        # Labels
        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
        ax.set_xlim([0, X.shape[0]*0.8])
        ax.set_ylim([-0.05, 1.05])
    
    # Visual aesthetics
    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)
    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize = 16, y = 1.03)
    fig.tight_layout()
    #fig.show()


def ModelComplexity(X, y):
    """ Calculates the performance of the model as model complexity increases. The learning and testing errors rates are then plotted. """ 
    
    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)

    # Vary the max_depth parameter from 1 to 10
    max_depth = np.arange(1,11)

    # Calculate the training and testing scores
    train_scores, test_scores = curves.validation_curve(DecisionTreeRegressor(), X, y,         param_name = "max_depth", param_range = max_depth, cv = cv, scoring = 'r2')

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    pl.figure(figsize=(7, 5))
    pl.title('Decision Tree Regressor Complexity Performance')
    pl.plot(max_depth, train_mean, 'o-', color = 'r', label = 'Training Score')
    pl.plot(max_depth, test_mean, 'o-', color = 'g', label = 'Validation Score')
    pl.fill_between(max_depth, train_mean - train_std,         train_mean + train_std, alpha = 0.15, color = 'r')
    pl.fill_between(max_depth, test_mean - test_std,         test_mean + test_std, alpha = 0.15, color = 'g')
    
    # Visual aesthetics
    pl.legend(loc = 'lower right')
    pl.xlabel('Maximum Depth')
    pl.ylabel('Score')
    pl.ylim([-0.05,1.05])
    #pl.show()


def PredictTrials(X, y, fitter, data):
    """ Performs trials of fitting and predicting data. """

    # Store the predicted prices
    prices = []

    for k in range(10):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y,             test_size = 0.2, random_state = k)
        
        # Fit the data
        reg = fitter(X_train, y_train)
        
        # Make a prediction
        pred = reg.predict([data[0]])[0]
        prices.append(pred)
        
        # Result
        print("Trial {}: ${:,.2f}".format(k+1, pred))

    # Display price range
    print("\nRange in prices: ${:,.2f}".format(max(prices) - min(prices)))


# In[ ]:


# Produce learning curves for varying training set sizes and maximum depths
ModelLearning(features, prices)


# ### Question 4 - Learning the Data
# * Choose one of the graphs above and state the maximum depth for the model. 
# * What happens to the score of the training curve as more training points are added? What about the testing curve? 
# * Would having more training points benefit the model? 
# 
# **Hint:** Are the learning curves converging to particular scores? Generally speaking, the more data you have, the better. But if your training and testing curves are converging with a score above your benchmark threshold, would this be necessary?
# Think about the pros and cons of adding more training points based on if the training and testing curves are converging.

# Answer:
# 
# A) max_depth = 1 (High Bias Scenario):  We see that initially the Testing Score(green line)  increases with increase in number of training points. But then it plateaus at a very low accuracy score of 0.4 or 40% and increase in number of training points have no effect. This shows that the model does not generalize well on unseen data. On the other hand, the Training Score(red line)  decreases with increase in the number of training points and gets saturated at a score of approximately 0.4 or 40%. This shows that the model is actually underfitting the data and is not complex enough. In this scenario, adding more training points will not benefit the model. Instead, its complexity should be increased for better fitting the dataset.
# 
# B)max_depth = 3 (Best scenario): Testing Score(green line) increases with increase in training points. It reaches a pretty high score of 0.8 and so we can see the model generalizes well. The Training Score(red line) decreases slightly and reaches 0.8 and stays constant. So we see it fits the model well and reaches a pretty high score. The testing score has two significant phases where the rates of change are different. One is the positive rate of change which goes on uptil approximately 200 training points (within this positive rate of change, we again observe two different rates. One is uptil 50 training points where the rate of increase is very high.The other is between 50 - 200 where the rate of increase is much lower.) and the other is the region where it plateaus with no/very little rate of change which is beyond 200 training points. So if we are below 200 training points, adding more training points will definitely improve the score but beyond that adding more training points will not be very useful as the rate plateaus.
# 
# C) max_depth = 6 (High Variance Scenario): Testing Score(green line) increases with increase in training points and reaches 0.7. Even though this is not a bad accuracy, it is not generalizing the data as well as max_depth = 3. The Training Score(red line) decrease ever so slightly and stays at 0.9 which is a big sign that it is overfitting the data. It is a High Variance problem. Here also, the testing score show a similar behaviour as the previous one (it plateaus after 200 training points). So once again, we will get an improvement in the testing score by adding more training points when the nuber of training points is less than 200, but after that adding more training points will not benefit us much.
# 
# D) max_depth = 10 (Higher Variance Scenario): Testing Score(green line) increases with increase in training points and reaches 0.7. So same problem as the previous one. It is not generalizing the data as well as scenario B). The  Training Score(red line) remains constant throughout showing a perfect accuracy of 100% or a score of 1 which tells us it is definitely overfitting the data. This is also a very High Variance problem. Once again the curve show exactly the same behaviour where adding more training points upto 200 will increase the score but not beyond that.
# 
# 

# ### Complexity Curves
# The following code cell produces a graph for a decision tree model that has been trained and validated on the training data using different maximum depths. The graph produces two complexity curves — one for training and one for validation. Similar to the **learning curves**, the shaded regions of both the complexity curves denote the uncertainty in those curves, and the model is scored on both the training and validation sets using the `performance_metric` function.  
# 
# ** Run the code cell below and use this graph to answer the following two questions Q5 and Q6. **

# In[ ]:


ModelComplexity(X_train, y_train)


# ### Question 5 - Bias-Variance Tradeoff
# * When the model is trained with a maximum depth of 1, does the model suffer from high bias or from high variance? 
# * How about when the model is trained with a maximum depth of 10? What visual cues in the graph justify your conclusions?
# 
# **Hint:** High bias is a sign of underfitting(model is not complex enough to pick up the nuances in the data) and high variance is a sign of overfitting(model is by-hearting the data and cannot generalize well). Think about which model(depth 1 or 10) aligns with which part of the tradeoff.

# Answer: We can easily recognize a problem related to High Bias or High Variance by simply looking at the graph of training and testing scores.
# 
# If there is High Bias, there will be very little gap between Training and Testing Scores. This is because in High Bias scenarios, the model underfits the data and also cannot generalize the data well resulting in both curves converging to a low score.
# 
# If there is High Variance, there will be a large gap between the Training and Testing Scores. This is because in High Variance model, even though the model fits well, it does not generalize well as a result of overfitting. This leads to a high Training Score but a relatively low Testing/Validation Score.
# 
# A) Maximum Depth = 1 (High Bias): Here both Training and Testing Scores are low. So the model is not fitting well and so it is not generalizing well. Thus the two curves are very close to each other and hence this is a High Bias situation.
# 
# B) Maximum Depth = 10 (High Variance): Here there is a huge gap between Training and Testing Scores. The Training score is almost perfect at 1, but the testing score is much low at around 0.7. So the model is overfitting and hence does not generalize well resulting in a lower Validation Score. So this is a High Variance situation with the curves being far apart. 

# ### Question 6 - Best-Guess Optimal Model
# * Which maximum depth do you think results in a model that best generalizes to unseen data? 
# * What intuition lead you to this answer?
# 
# ** Hint: ** Look at the graph above Question 5 and see where the validation scores lie for the various depths that have been assigned to the model. Does it get better with increased depth? At what point do we get our best validation score without overcomplicating our model? And remember, Occams Razor states "Among competing hypotheses, the one with the fewest assumptions should be selected."

# Answer:  Maximum Depth = 4
# 
# The validation score seems to plateau here. So this is the highest validation score we can get i.e best generalization of unseen data.
# 
# The gap between the Training Score and the Validation Score is not significantly large here too which indicates a High Variance Situation.

# -----
# 
# ## Evaluating Model Performance
# In this final section of the project, we will construct a model and make a prediction on the client's feature set using an optimized model from `fit_model`.

# ### Question 7 - Grid Search
# * What is the grid search technique?
# * How it can be applied to optimize a learning algorithm?
# 

# **Answer: ** The Grid search technique allows us to define a grid of the hyperparameters for a specific classifier and then the Grid search technique exhaustively tries out every possible combinations of the hyperparameters values in order to find the best model. After that we can use cross validation techniques like K-fold cross validation or Stratified Shuffle Split to find the highest accuracy by using the hyperparameters suggested by Grid Search technique optimizing the learning algorithm.
# 
# ** Point to Note: ** Due to its exhaustive search nature, grid search can be computationally expensive, especially when data size is large and model is complicated. Sometimes we resort to randomized search in this case to search only some combinations of the parameters. 
# (http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.RandomizedSearchCV.html#sklearn-grid-search-randomizedsearchcv)

# ### Question 8 - Cross-Validation
# 
# * What is the k-fold cross-validation training technique? 
# 
# * What benefit does this technique provide for grid search when optimizing a model?
# 
# **Hint:** When explaining the k-fold cross validation technique, be sure to touch upon what 'k' is, how the dataset is split into different parts for training and testing and the number of times it is run based on the 'k' value.
# 
# When thinking about how k-fold cross validation helps grid search, think about the main drawbacks of grid search which are hinged upon **using a particular subset of data for training or testing** and how k-fold cv could help alleviate that. You can refer to the [docs](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation) for your answer.

# **Answer: ** In K-fold cross validation technique, we partition the data into k-bins of equal size. After that we run k separate learning experiments. In each of those, we pick one of the k subsets as our testing set. The remaining k-1 bins are put together into the training set. Then we train our machine learning algorithm and just like before test the performance on the testing set. The key thing in cross validation is we run this multiple times (k times) and then we average the k different testing set performances for the k different hold out sets. So we average the test results from those k experiments. So obviously this takes more computation time as now we have to run k separate learning experiments, but the assessment of the learning algorithm will be more accurate.
# 
# If we run Grid Search without running a cross validation set, we will have different sets of optimal hyperparameters because without a cross validation set, the estimate of out-of-sample performance would have a high variance. 
# 
# So in summary, without k-fold cross validation, the Grid Search will select hyper parameter values which works really well on the sample train test split data but there is a high risk that it will work poorly for unknown datasets because of high variance.
# 
# 

# ### Implementation: Fitting a Model
# Our final implementation requires that we bring everything together and train a model using the **decision tree algorithm**. To ensure that we are producing an optimized model, we will train the model using the grid search technique to optimize the `'max_depth'` parameter for the decision tree. The `'max_depth'` parameter can be thought of as how many questions the decision tree algorithm is allowed to ask about the data before making a prediction. Decision trees are part of a class of algorithms called *supervised learning algorithms*.
# 
# In addition, we will find our implementation is using `ShuffleSplit()` for an alternative form of cross-validation (see the `'cv_sets'` variable). While it is not the K-Fold cross-validation technique you describe in **Question 8**, this type of cross-validation technique is just as useful!. The `ShuffleSplit()` implementation below will create 10 (`'n_splits'`) shuffled sets, and for each shuffle, 20% (`'test_size'`) of the data will be used as the *validation set*. While we're working on our implementation, we'll think about the contrasts and similarities it has to the K-fold cross-validation technique.
# 
# Please note that ShuffleSplit has different parameters in scikit-learn versions 0.17 and 0.18.
# For the `fit_model` function in the code cell below, we will need to implement the following:
# - Use [`DecisionTreeRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) from `sklearn.tree` to create a decision tree regressor object.
#   - Assign this object to the `'regressor'` variable.
# - Create a dictionary for `'max_depth'` with the values from 1 to 10, and assign this to the `'params'` variable.
# - Use [`make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html) from `sklearn.metrics` to create a scoring function object.
#   - Pass the `performance_metric` function as a parameter to the object.
#   - Assign this scoring function to the `'scoring_fnc'` variable.
# - Use [`GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) from `sklearn.grid_search` to create a grid search object.
#   - Pass the variables `'regressor'`, `'params'`, `'scoring_fnc'`, and `'cv_sets'` as parameters to the object. 
#   - Assign the `GridSearchCV` object to the `'grid'` variable.

# In[ ]:


# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a decision tree regressor trained on the input data [X, y]. """ 
    
    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # TODO: Create a decision tree regressor object
    
    regressor = DecisionTreeRegressor(random_state = 1001)

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    tree_range = range(1, 11)
    params = dict(max_depth=[1,2,3,4,5,6,7,8,9,10])

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    grid = GridSearchCV(regressor,params,scoring=scoring_fnc,cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


# ### Making Predictions
# Once a model has been trained on a given set of data, it can now be used to make predictions on new sets of input data. In the case of a *decision tree regressor*, the model has learned *what the best questions to ask about the input data are*, and can respond with a prediction for the **target variable**. We can use these predictions to gain information about data where the value of the target variable is unknown — such as data the model was not trained on.

# ### Question 9 - Optimal Model
# 
# * What maximum depth does the optimal model have? How does this result compare to your guess in **Question 6**?  
# 
# Run the code block below to fit the decision tree regressor to the training data and produce an optimal model.

# In[ ]:


# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))


# ** Hint: ** The answer comes from the output of the code snipped above.
# 
# **Answer: ** The optimum model has a maximum depth of 4. This exactly matches our guess from ** Question 6 **. Both results are reliable as in both cases, we did cross validation with Shufflesplit combined with checking against a range of the max_depth hyperparamters to give us the most optimal value of the max_depth. So based on our course of action, there is very little chance that our model will work poorly for unknown datasets because of high variance.

# ### Question 10 - Predicting Selling Prices
# Imagine that we were a real estate agent in the Boston area looking to use this model to help price homes owned by our clients that they wish to sell. We have collected the following information from three of our clients:
# 
# | Feature | Client 1 | Client 2 | Client 3 |
# | :---: | :---: | :---: | :---: |
# | Total number of rooms in home | 5 rooms | 4 rooms | 8 rooms |
# | Neighborhood poverty level (as %) | 17% | 32% | 3% |
# | Student-teacher ratio of nearby schools | 15-to-1 | 22-to-1 | 12-to-1 |
# 
# * What price would you recommend each client sell his/her home at? 
# * Do these prices seem reasonable given the values for the respective features? 
# 
# **Hint:** Use the statistics you calculated in the **Data Exploration** section to help justify your response.  Of the three clients, client 3 has has the biggest house, in the best public school neighborhood with the lowest poverty level; while client 2 has the smallest house, in a neighborhood with a relatively high poverty rate and not the best public schools.
# 
# Run the code block below to have your optimized model make predictions for each client's home.

# In[ ]:


# Produce a matrix for client data
client_data = [[5, 17, 15],  [4, 32, 22],  [8, 3, 12]]   

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))


# ### Visualization

# In[ ]:


from matplotlib import pyplot as plt

clients = np.transpose(client_data)
pred = reg.predict(client_data)
for i, feat in enumerate(['RM', 'LSTAT', 'PTRATIO']):
    plt.scatter(features[feat], prices, alpha=0.25, c=prices)
    plt.scatter(clients[i], pred, color='black', marker='x', linewidths=2)
    plt.xlabel(feat)
    plt.ylabel('MEDV')
    print()


# **Answer: **
# 
# Client 1: $403,025.00
# 
# Client 2: $237,478.72
# 
# Client 3: $931,636.36
# 
# In our initial ** Data Exploration ** section, we saw that the price is positively correlated with the number of rooms and negatively correlated with Neighbourhood Poverty level and Student-teacher ratio of nearby schools. Also these were the statistics of our data.
# 
# Minimum price: $105,000.00
# 
# Maximum price: $1,024,800.00
# 
# Mean price: $454,342.94
# 
# Median price $438,900.00
# 
# Standard deviation of prices: $165,340.28
# 
# So we see that for Client 1 and 2, the price of the house is below the median price of the houses. This is reasonable because of 
# 
# a) High Poverty Level and Student to Teacher ratio for client 2.
# 
# b) Average Poverty level and Student to Teacher ratio for client 1.
# 
# For Client 3, we see that the price is well over the median house price and very close to the maximum house price. This is also reasonable because of very low Poverty Level and Student to Teacher ratio and also a high number of rooms.
# 
# So overall, the prices for all the clients seem reasonable.

# ### Perfomance Metric
# 
# Let us calculate the R squared value for our model.

# In[ ]:


reg = fit_model(X_train, y_train)
pred = reg.predict(X_test)
score = performance_metric(y_test,pred)
print("R Squared Value: " + str(score))


# So we get a pretty good R squared score from our model.

# ### Visualization

# In[ ]:


import matplotlib.pyplot as plt
plt.hist(prices, bins = 20)
for price in reg.predict(client_data):
    plt.axvline(price, lw = 5, c = 'r')


# ### Sensitivity
# An optimal model is not necessarily a robust model. Sometimes, a model is either too complex or too simple to sufficiently generalize to new data. Sometimes, a model could use a learning algorithm that is not appropriate for the structure of the data given. Other times, the data itself could be too noisy or contain too few samples to allow a model to adequately capture the target variable — i.e., the model is underfitted. 
# 
# **Run the code cell below to run the `fit_model` function ten times with different training and testing sets to see how the prediction for a specific client changes with respect to the data it's trained on.**

# In[ ]:


PredictTrials(features, prices, fit_model, client_data)


# ### Question 11 - Applicability
# 
# * In a few sentences, discuss whether the constructed model should or should not be used in a real-world setting.  
# 
# **Hint:** Take a look at the range in prices as calculated in the code snippet above. Some questions to answering:
# - How relevant today is data that was collected from 1978? How important is inflation?
# - Are the features present in the data sufficient to describe a home? Do you think factors like quality of apppliances in the home, square feet of the plot area, presence of pool or not etc should factor in?
# - Is the model robust enough to make consistent predictions?
# - Would data collected in an urban city like Boston be applicable in a rural city?
# - Is it fair to judge the price of an individual home based on the characteristics of the entire neighborhood?

# **Answer: ** 
# 
# 1) The data which was collected in 1978 is not so relevant today because demographics and economy has changed a lot since then.
# 
# 2) The features present in the data is not sufficient to describe a home. There are only three features present right now. We can add more features like crime rate, transportation avalibility, presence of pool or not, square feet of the plot area, quality of appliances, flooring in the home and more.
# 
# 3) This model based on its current feature is robust enough to make consistent predictions with a small margin of error.
# 
# 4) Data collected in an urban city like Boston may not be applicable in a rural city as many properties will change like the Demographics, Economy, Average income etc. So we would have to take in account a lot of other features in order to build an effective model
# 
# 5) Neighbourhood plays a very vital role in judging the price of a house like the crime rate, schools, transportation etc. But if an individual house has some marked characteristics which can overshadow the factors that neighbourhood plays, then it would not be fair to judge the price of an individual home based on the characteristics of the entire neighborhood.
