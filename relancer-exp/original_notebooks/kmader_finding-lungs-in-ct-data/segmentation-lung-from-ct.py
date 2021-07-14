#!/usr/bin/env python
# coding: utf-8

# The code "fork"/(type by hand) from Joshy Cyriac's [cnn with keras](https://www.kaggle.com/irrwitz/cnn-with-keras)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../../../input/kmader_finding-lungs-in-ct-data/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../../../input/kmader_finding-lungs-in-ct-data"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
from glob import glob
import matplotlib.pyplot as plt

import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,                          Flatten, Convolution2D, MaxPooling2D,                          BatchNormalization, UpSampling2D
from keras.utils import np_utils

from skimage.io import imread
from sklearn.model_selection import train_test_split


# In[ ]:


K.image_data_format()


# In[ ]:


# wired ? tf: channels_last, th: channels_first
K.set_image_dim_ordering('th')


# In[ ]:


# 4 表示步长
jimread = lambda x: np.expand_dims(imread(x)[::4,::4],0)


# In[ ]:


BASE_IMAGE_PATH = '../../../input/kmader_finding-lungs-in-ct-data'
all_images = glob(os.path.join(BASE_IMAGE_PATH, '2d_images', '*.tif'))
# 不知道代码意思就通过print查看
# print('_masks'.join(all_images[0].split('_images')))
all_masks = ['_masks'.join(c_file.split('_images')) for c_file in all_images]


# In[ ]:


print(len(all_masks), 'matching files found')


# In[ ]:


print(jimread(all_images[0]).shape)


# In[ ]:


test_image = jimread(all_images[0])
test_mask = jimread(all_masks[0])
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(test_image[0])
ax2.imshow(test_mask[0])


# In[ ]:


print('Total samples are', len(all_images))
print('Image resolution is', test_image.shape)


# 读取所有的图片，并转成numpy arrays
# 

# In[ ]:


images = np.stack([jimread(image) for image in all_images], 0)
# 为啥只对masks normalization 处理
masks = np.stack([jimread(image) for image in all_masks], 0) /255.0
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.1)
print('Training input is ', X_train.shape)
print('Training output is {}, min is {}, max is {}'.format(y_train.shape, y_train.min(), y_train.max()))
print('Testing set is ', X_test.shape)


# In[ ]:


print(images.shape[1:])


# In[ ]:


# Create a deep nn
model = Sequential()
model.add(Convolution2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=images.shape[1:], padding='same' )) 
model.add(Convolution2D(filters=64, kernel_size=(3,3), activation='sigmoid', input_shape=images.shape[1:], padding='same' )) 
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dense(64, activation='relu'))
model.add(Convolution2D(filters=1, kernel_size=(3,3), activation='sigmoid', input_shape=images.shape[1:], padding='same' )) 
model.add(UpSampling2D(size=(2,2)))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy','mse']) 
print(model.summary())


# In[ ]:


history = model.fit(X_train, y_train, validation_split=0.10, epochs=10, batch_size=10)


# In[ ]:


history.history.keys()


# In[ ]:


fig, ax = plt.subplots(1, 2)
ax[0].plot(history.history['acc'], 'b')
ax[0].set_title('Accuraccy')
ax[1].plot(history.history['loss'], 'r')
ax[1].set_title('Loss')


# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.imshow(X_test[0,0])
ax2.imshow(y_test[0,0])
ax3.imshow(model.predict(X_test)[0,0])


# In[ ]:




