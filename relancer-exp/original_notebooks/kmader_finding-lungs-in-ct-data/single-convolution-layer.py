#!/usr/bin/env python
# coding: utf-8

# 

# 

# In[ ]:


import os
import numpy as np
np.random.seed(123)
import pandas as pd

from glob import glob
import matplotlib.pyplot as plt

import keras.backend as K
from keras.models import Sequential
from keras.layers import Convolution2D

from skimage.io import imread
from sklearn.model_selection import train_test_split


# In[ ]:


# set channels first notation
K.set_image_dim_ordering('th')


# In[ ]:


# helper function to load the image and downsample it by 4
jimread = lambda x: np.expand_dims(imread(x)[::4, ::4], 0)


# In[ ]:


BASE_IMAGE_PATH = '../../../input/kmader_finding-lungs-in-ct-data'
all_images = glob(os.path.join(BASE_IMAGE_PATH, '2d_images', '*.tif'))
all_masks = ['_masks'.join(c_file.split('_images')) for c_file in all_images]
print(len(all_masks), 'matching files found')


# 

# In[ ]:


test_image = jimread(all_images[0])
test_mask = jimread(all_masks[0])
fig, (ax1 ,ax2) = plt.subplots(1, 2)
ax1.imshow(test_image[0])
ax1.set_title('Lung slice')
ax2.imshow(test_mask[0])
ax2.set_title('Ground truth')


# In[ ]:


print('Total samples are', len(all_images))
print('Image resolution is', test_image.shape)


# 

# In[ ]:


images = np.stack([jimread(i) for i in all_images], 0) / 1024.0
masks = np.stack([jimread(i) for i in all_masks], 0) / 255.0
X_train, X_test, y_train,  y_test = train_test_split(images, masks, test_size=0.2)
print('Training input is', X_train.shape)
print('Training output is {}, min is {}, max is {}'.format(y_train.shape, y_train.min(), y_train.max()))
print('Testing set is', X_test.shape)


# 

# In[ ]:


def evaluate(kernel_size=(1,1)):
    model = Sequential()
    model.add(Convolution2D(filters=1, kernel_size=kernel_size, activation='relu', input_shape=images.shape[1:], padding='same' )) 
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy', 'mse']) 
    loss_history = []
    print(model.summary())
    loss_history += [model.fit(X_train, y_train, validation_split=0.20, epochs=10, batch_size=10)]
    return model, loss_history


# 

# In[ ]:


model0, loss_history = evaluate(kernel_size=(1,1))
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.imshow(X_test[0,0])
ax2.imshow(y_test[0,0])
ax3.imshow(model0.predict(X_test)[0,0])


# 

# In[ ]:


model1, loss_history = evaluate(kernel_size=(3,3))
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.imshow(X_test[0,0])
ax2.imshow(y_test[0,0])
ax3.imshow(model1.predict(X_test)[0,0])


# In[ ]:


model2, loss_history = evaluate(kernel_size=(30,30))
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.imshow(X_test[0,0])
ax2.imshow(y_test[0,0])
ax3.imshow(model2.predict(X_test)[0,0])

