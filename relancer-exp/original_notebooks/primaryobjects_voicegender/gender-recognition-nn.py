#!/usr/bin/env python
# coding: utf-8

# # Gender Recognition from voice using Deep Learning And Neural Networks

# # **** AIM****
# 1. To build neural networks to classify the gender of the voice and maximise the accuracy of the model
# 2. To compare the accuracy of Deep learning-Neural Network model with machine learning classifiers

# Before we dive in let me give a brief of what we are upto. We have a dataset which based on certain paramaters classifies a voice based on gender. How do humans do it?
# 
#     Sound waves travel into the ear canal until they reach the eardrum. The eardrum passes the vibrations through the middle ear bones or ossicles into the inner ear. The inner ear is shaped like a snail and is also called the cochlea. Inside the cochlea, there are thousands of tiny hair cells. Hair cells change the vibrations into electrical signals that are sent to the brain through the hearing nerve. The brain tells you that you are hearing a sound and what that sound is.
# 
# What happens in the brain is neurons perform certain operations to classify the sound, this is exactly what we will be trying to simulate. 

# In[ ]:


import os
for dirname, _, filenames in os.walk("../../../input/primaryobjects_voicegender"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Step 1
# Import the libraries
# 1. matplotlib :: To plot graphs 
# 2. numpy :: To perform operations and manipulate arrays 
# 3. pandas :: To read and manage the data from the file
# 4. Import ML basic classification models :: from sklearn for classification
# 5. Import Neural network building libraries :: from keras

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from mpl_toolkits.mplot3d import Axes3D

#Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

#plotting missing data
import missingno as msno

#classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

#Neural network building libraries
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import History 
from keras.utils import plot_model
from keras.optimizers import SGD


# # Step 2
# Loading the dataset and performing EDA (Exploratory Data Analysis) over the dataset. 
# To analyse and understand the dataset, its features and target classes

# In[ ]:


voice=pd.read_csv("../../../input/primaryobjects_voicegender/voice.csv")
voice.head(5)


# In[ ]:


print("\n",voice.info())


# In[ ]:


voice.describe()


# In[ ]:


#visualizing no missing value.
msno.matrix(voice)


# **Shows no null values so cleaning not required**

# In[ ]:


#creating a copy
data=voice.copy()


# In[ ]:


# Distribution of target varibles
colors = ['pink','Lightblue']
df = data[data.columns[-1]]
plt.pie(df.value_counts(),colors=colors,labels=['female','male'])
plt.axis('equal')
print (data['label'].value_counts())


# In[ ]:


#Radviz circle 
#Good to compare every feature
pd.plotting.radviz(data,"label")


# In[ ]:


# Pairplotting


# In[ ]:


data.drop('label' ,axis=1).hist(bins=30, figsize=(12,12))
pl.suptitle("Histogram for each numeric input variable")
print()


# In[ ]:


#corelation matrix.
cor_mat= data[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(15,15)


# In this section the corelation between different features is analyzed. 'Heat map' is plotted which clearly visulizes the corelation between different features

# # Step 3
# 
# Now since we have the feature set and the set of dependent variables, We observe that the 'label' has strings and in maths we need values so  we will convert it to numerical values Male=1 and Female=0

# In[ ]:


# Convert string label to float : male = 1, female = 0
dict = {'label':{'male':1,'female':0}}      # label = column name
data.replace(dict,inplace = True)           # replace = str to numerical
x = data.loc[:, data.columns != 'label']
y = data.loc[:,'label']


# # Step 4
# We need to separate the dependent and independent variables. Here the first 20 set columns consists of the features and the last coloumn is the dependent variable, which takes two integer values i.e 1 (Male) and 0 (Female)
# 
# X as feature columns and Y as dependent column

# In[ ]:


array = data.values
X = array[:,0:20]
Y = array[:,20]


# # Step 5
# 
# Divide the data into training set and test set, One set to train the neural Network and the other set to test the neural network. 

# In[ ]:


X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)


# # Step 6 
# **Scaling**
# 
# 
# Now if we observe the values in various coloumns we see that there is a problem either the values are extremely close to zero or the all the coloumn are of not the same scale.  there is a lot of times where we will need to calculate slopes assume in the denominator two point are really close to zero, subtracting will lead it much more closer to zero and the slope assumes an amazingly huge value, so to prevent this kind of problems we generally use scaling in Neural Networks.

# In[ ]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Step 7
# Building **different Machine Learning classifiers** and finding the accuracy score of each model.
# 
# Also to find the accurate model with highest accuracy

# In[ ]:


#Appending different Models to a list

models = []

models.append(( 'LR ', LogisticRegression()))
models.append(( 'SVC', SVC(kernel='linear', C=1.0, random_state=0)))
models.append(( 'LDA', LinearDiscriminantAnalysis()))
models.append(( 'KNN', KNeighborsClassifier(n_neighbors=20, p=2, metric='minkowski')))
models.append(( 'CLF', DecisionTreeClassifier(criterion="entropy",max_depth=3)))
models.append(( 'RFC', RandomForestClassifier(max_depth=2, random_state=0)))
models.append(( 'MLP', MLPClassifier(hidden_layer_sizes=(3,3),max_iter=3000, activation = 'relu',solver='adam',random_state=1)))
models.append(( 'GNB', GaussianNB()))


# In[ ]:


#Finding Mean Accuracy for Models

results = []
names = []
meanscore=[]
scoring = 'accuracy'

for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: (%f)" % (name, cv_results.mean()*100)
    meanscore.append(cv_results.mean()*100)
    print("Mean Accuracy score", msg)

print("\nHighest Mean Accuracy is for the classifer LDA", max(meanscore))
plt.plot(names,meanscore,marker='o')
plt.xlabel('Models')
plt.ylabel('Model Accuracy')
plt.title('ML classifiers and Accuracy score',size=25)


# # Step 8
# 
# Now starts the actual  building of the neural network using Keras.
# 
# Before we get into the code let us try to understand the neural network structure we are aiming to build. Here in the data set there are 20 paramaters which can also be called features and these are fed to the nodes on a one-to-one basis that is one node recieves one input. We will call this the first layer and this is what this piece of code does.
# 
# The Dense is used to specify the fully connected layer.
# 
#     classifier.add(Dense(output_dim=16,init='uniform',activation='relu',input_dim=20))
# 
# next we pass this sound to the processing unit the brain where we have a lot of itermediate processing neurons before we actually get the output. In this case we will add just 2 intermediate stages of processing neurons with 16 nodes in each layer.
# 
#     classifier.add(Dense(output_dim=16,init='uniform',activation='relu'))
# 
#     classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
# 
# Now we need to get the output and one node will do the job
# 
#     classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
# 
# If you are wondering what is relu and sigmoid well these are the functions which are used to calculate the weights/loss etc.
# 
# Now we need to specify the loss function and the optimizer. It is done using compile function in keras.
# 
#     classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# In the end we feed out data to the neural Network and wait for the magic to happen.

# In[ ]:


classifier=Sequential()
history = History()

#number of input variables = 20 so input_dim is only for the first layer
classifier.add(Dense(output_dim=16,init='uniform',activation='relu',input_dim=20)) #first layer
classifier.add(Dense(output_dim=16,init='uniform',activation='relu'))   #first Hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))    #Second Hidden layer

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid')) #output layer

#Running the artificial neural network
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.summary()


# # Step 9
# **Training model :**
# Now we are done with building a neural network and we will train it.Training step is simple in keras. 
# 
#        ( classifier.fit)  is used to train it.
# It is always important to see, what actually is happening and how the model is learning. So with every epoch there is some learning which happens. The model is capable of calculating the loss it is facing from the actual result and then correspondingly adjusts its weight in automatically

# In[ ]:


trained=classifier.fit(X_train,Y_train,batch_size=5,epochs=20,validation_split=0.2,callbacks=[history],shuffle=2)


# # Step 10 (Final Step)
# 
# Now we can check the model’s performance on test data:

# In[ ]:


y_pred=classifier.predict(X_train)
y_pred = np.round(y_pred)

print('Accuracy by the Neural Network on train dataset is',metrics.accuracy_score(y_pred,Y_train)*100,'%')

y_pred=classifier.predict(X_test)
y_pred = np.round(y_pred)

print('Accuracy by the Neural Network on test dataset is ',metrics.accuracy_score(y_pred,Y_test)*100,'%')


# # Visualization of model accuracy

# In[ ]:


plt.plot(history.history['loss'], color = 'red',label='Variaton Loss over the epochs',)
plt.plot(history.history['accuracy'],color='green',label='Variation in Accuracy over the epochs')

plt.xlabel('Epochs')
plt.title('Loss/Accuracy VS Epoch on test Dataset using our model')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='best')
print()


# If we observe the graph, Over a period of time it clear that the loss is gradually hitting zero and the Accuracy is increasing at a considerable rate.

# In[ ]:


plt.plot(trained.history['accuracy'])
plt.plot(trained.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
print()

plt.plot(trained.history['loss'])
plt.plot(trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
print()


# In[ ]:


plt.plot(Y_test[-30:],linestyle='--',label='Actual value',linewidth=3,marker='o' ,markerfacecolor='green',markersize=15,color='green')
plt.plot(y_pred[-30:],linestyle='-.',label='Predicted value',linewidth=3,marker='o' ,markerfacecolor='red',markersize=10,color='red')
plt.title('Validating the Model for 30 voices',size=15)
plt.xlabel("Voice notes")
plt.ylabel("Male(1)/ female(0)")
plt.legend(loc='center left')


# The actual and predicted value is visualized

