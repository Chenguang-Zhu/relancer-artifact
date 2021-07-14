#!/usr/bin/env python
# coding: utf-8

# # 1. Kernel objectives
# 
# This kernel is created for 2 reasons:
# 
# 1. Train CNN to recognize subspecies of Bee
# 
# 2. Feed <b>my photo</b> to trained CNN and determine what subspecies of Bee I actually am :)
# 

# In[ ]:


############## Necessary imports #################
import pandas as pd

########################
# Common 
########################
import sys
import random
import gc
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

########################
##### Image processing
########################
import imageio
import skimage
import skimage.io
import skimage.transform
#from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
import scipy

########################
# Plotting
########################
import matplotlib.pyplot as plt
import seaborn as sns

########################
# ML libs
########################
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
import tensorflow

########################
# Global variables and settings
########################
img_folder="../../../input/jenny18_honey-bee-annotated-images/bee_imgs/bee_imgs/"
img_width=100
img_height=100
img_channels=3

# Set NumPy and TensorFlow random seed to make results reproducable
np.random.seed(42)
tensorflow.set_random_seed(2)


# # 2. Read the data

# In[ ]:


bees=pd.read_csv("../../../input/jenny18_honey-bee-annotated-images/bee_data.csv", index_col=False, parse_dates={'datetime':[1,2]}, dtype={'subspecies':'category', 'health':'category','caste':'category'}) 

def read_or_skip(file):
    """This function is to supress imageio exception if file doesn't exist"""
    try:
        img = skimage.io.imread(img_folder + file)
        img = skimage.transform.resize(img, (img_width, img_height), mode='reflect')
        return img[:,:,:img_channels]
    except:
        #print('Skipping %s. %s' %(file, sys.exc_info()[1]))
        return None

bees['img'] = bees['file'].apply(read_or_skip)
bees.dropna(inplace=True)

# Print sample data without img array
bees.drop('img',axis=1).head()

plt.style.use('seaborn')


# # 3. Bee data EDA
# <b>Subspecies</b> column is of interest for this kernel's goal.
# ## 3.1 Distribution of bees by subspecies
# Check how much Bees we have in each subspecies

# In[ ]:


# Plot count by subspecies
bees.subspecies.value_counts().plot.bar(title="Subspecies count in dataset")
plt.ylabel("Count")
print()
bees.subspecies.value_counts()


# It is really unbalanced, need to figure it out before we feed this to CNN.

# # 4. Data preprocessing
# ## 4.1. Balance by subspecies 
# Let's make the count of each subspecies to be equal. It seems there is no standard function to do <b>balanced</b> train_test_split. But <b>sklearn.utils.resample</b> will do the job for each subspecies category separately, in a loop. We don't worry about upsampling duplicates, hoping that later image augmentation will vary the input enough. But we should split all bees to train and test <b>before balancing</b> to prevent upsampled duplicates occur in both train and test.

# In[ ]:


# The plan
# 1. Split all bees to train and test subsets, unbalanced.
# 2. Balance train and test subsets separately by subspecies categories
# 3. Extract features and labels from balanced train and balanced test datasets. 
# The data is prepared to CNN now.

# 1. Split bees considering train/test ratio. Labels are kept in features 
# Ignore labels output from train_test_split, we'll need to balance train/test data 
# before getting labels
train_bees_unbalanced, test_bees_unbalanced, _train_labels_unbalanced, _test_labels_unbalanced = train_test_split(bees, bees.subspecies)
# Delete not needed data to avoid memory error
del _train_labels_unbalanced
del _test_labels_unbalanced

# 2. Balance train and test subsets separately by subspecies categories.

# Set variables
# Subspecies categories for rebalancing by them
ss_names = train_bees_unbalanced.subspecies.values.unique() 
ss_num = ss_names.size
# Total rows in rebalanced dataset. Can be lower or higher than original data rows.
n_samples = bees.size / 2
ratio = 0.25

# Train/test rows nums
test_num = n_samples * ratio
train_num = n_samples - test_num

# Resample each subspecies category and add to resulting train dataframe
train_bees_balanced = pd.DataFrame()
test_bees_balanced = pd.DataFrame()
for ss in ss_names:
    # Resample category in train bees
    bees_cur = train_bees_unbalanced[train_bees_unbalanced.subspecies == ss]
    bees_cur_resampled = resample(bees_cur, n_samples=int(train_num/ss_num))
    train_bees_balanced = pd.concat([train_bees_balanced, bees_cur_resampled])
    # Resample category in test bees
    bees_cur = test_bees_unbalanced[test_bees_unbalanced.subspecies == ss]
    bees_cur_resampled = resample(bees_cur, n_samples=int(test_num/ss_num))
    test_bees_balanced = pd.concat([test_bees_balanced, bees_cur_resampled])

# Delete not needed data to avoid memory error
del train_bees_unbalanced
del test_bees_unbalanced

# 3. Extract features and labels from balanced train, test

# Get train features and labels from train rebalanced bees
train_labels = pd.get_dummies(train_bees_balanced.subspecies)
train_data=np.stack(train_bees_balanced.img)

# Get test features and one hot encoded labels from balanced test
test_labels = pd.get_dummies(test_bees_balanced.subspecies)
test_data = np.stack(test_bees_balanced.img)
    

# Plot resampled data to check
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,3))
train_bees_balanced.subspecies.value_counts().plot.bar(title ="Balanced train subspecies", ax=ax[0])
ax[0].set_ylabel("Count")
test_bees_balanced.subspecies.value_counts().plot.bar(title ="Balanced test subspecies", ax=ax[1])
ax[1].set_ylabel("Count")

print()

# Delete not needed data to avoid memory error
del train_bees_balanced
del test_bees_balanced

gc.collect()


# ## 5. Create and train CNN

# In[ ]:


# Data augmentation - rotate, zoom and shift input images.
generator = ImageDataGenerator( featurewise_center=False,   samplewise_center=False,   featurewise_std_normalization=False,   samplewise_std_normalization=False,   zca_whitening=False,   rotation_range=180,   zoom_range = 0.1,  width_shift_range=0.1,   height_shift_range=0.1,   horizontal_flip=True,   vertical_flip=True)   
generator.fit(train_data)

# Split train data to features and labels
train_data, train_data_val, train_labels, train_labels_val = train_test_split(train_data, train_labels, test_size=0.1) 
# Build and train CNN model
model = Sequential()
model.add(Conv2D(6, kernel_size=3, input_shape=(img_width, img_height,3), activation='relu'))
model.add(MaxPool2D(2))
model.add(Conv2D(12, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(train_labels.columns.size, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# We'll stop training if no improvement after some epochs
earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
# Save the best model during the traning
checkpointer = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)

# Train
training = model.fit_generator(generator.flow(train_data,train_labels, batch_size=100), epochs = 30, validation_data=(train_data_val, train_labels_val), steps_per_epoch=100,   callbacks=[earlystopper, checkpointer]) 

# Load the best model
model.load_weights('best_model.h5')


# ## 6. Evaluate trained model

# In[ ]:


## Trained model analysis and evaluation
f, ax = plt.subplots(2,1, figsize=(5,5))
ax[0].plot(training.history['loss'])
ax[0].set_title('Detect kind of Bee: loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')

# Accuracy
ax[1].plot(training.history['acc'])
ax[1].set_title('Detect kind of Bee: accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
plt.tight_layout()
print()

# Accuracy by subspecies
test_pred = model.predict(test_data)
acc_by_subspecies = np.logical_and((test_pred > 0.5), test_labels).sum()/test_labels.sum()
acc_by_subspecies.plot(kind='bar', title='Subspecies prediction accuracy')
plt.ylabel('Accuracy')
print()

# Loss function and accuracy
test_res = model.evaluate(test_data, test_labels)
print('Evaluation: loss function: %s, accuracy:' % test_res[0], test_res[1])


# > # 7. Who am I?
# Now I am coming to the very end of my research and can apply the whole power of modern artificial intelligence to classify myself.

# In[ ]:


# Load my Kaggle avatar
my_img_url = 'https://storage.googleapis.com/kaggle-avatars/images/701733-kg.jpg'
my_img_full = skimage.io.imread(my_img_url)

# Prepare image for prediction
my_img = skimage.transform.resize(my_img_full, (img_width, img_height), mode='reflect')[:,:,:img_channels]
# Predict my subspecies with already well-trained CNN
my_pred_index = model.predict(my_img[None,...]).argmax()
my_subspecies = test_labels.columns[my_pred_index]

# Use default style wo grid lines
plt.style.use('default')

# Draw the photo titled by subspecies recognized
plt.figure(figsize=(2,2))
plt.imshow(my_img_full)
plt.title(my_subspecies)
print()

