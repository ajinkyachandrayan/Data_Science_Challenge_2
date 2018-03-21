# coding: utf-8

# In[1]:

get_ipython().magic('pylab inline')
import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score

import tensorflow as tf
import keras


# In[2]:

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)


# In[3]:

import glob
from PIL import Image
import os.path
import re
from resizeimage import resizeimage


# In[4]:

data_dir = os.path.abspath('images_35_35/')


# In[5]:

os.path.exists(data_dir)


# In[6]:

train = pd.read_csv('train.csv',header = 0, sep = ',')
test = pd.read_csv('test.csv',header = 0, sep = ',')

sample_submission = pd.read_csv('sample_submission.csv', header = 0, sep = ",")

train.head()


# In[7]:

train.orientation.unique()


# In[8]:

train.shape


# In[9]:

img_name = rng.choice(train['id'])

img = imread('images_35_35/%d.jpg' % img_name, flatten=True)

pylab.imshow(img, cmap = 'gray')
pylab.axis('off')
pylab.show()


# In[10]:

img


# In[11]:

##  Reading Train & Test


# In[12]:

temp = []
for img_name in train['id']:
    img = imread('images_35_35/%d.jpg' % img_name, flatten=True)
    temp.append(img)
    
train_x = np.stack(temp)


# In[13]:

train_x = train_x.reshape(9980, 35, 35, 1)
train_x = train_x.astype('float32')
train_x /= 255


# In[14]:

temp = []
for img_name in test['id']:
    img = imread('images_35_35/%d.jpg' % img_name, flatten=True)
    temp.append(img)
    
test_x = np.stack(temp)


# In[15]:

test_x = test_x.reshape(4124, 35, 35, 1)
test_x = test_x.astype('float32')
test_x /= 255


# In[16]:

train_y = train['orientation']


# In[17]:

train_y.head()


# In[18]:

num_classes = 4


# In[19]:

train_y = train_y -1


# In[20]:

train_y = keras.utils.to_categorical(train_y, num_classes)


# In[21]:

train_y


# In[22]:

## Splitting Train set into Train and Validation


# In[23]:

split_size = int(train_x.shape[0]*0.8)


# In[24]:

train_xx, val_x = train_x[:split_size], train_x[split_size:]


# In[25]:

train_yy, val_y = train_y[:split_size], train_y[split_size:]


# In[26]:

from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator


# In[27]:

from keras.applications.vgg19 import VGG19


# In[194]:

# 1. Convolution
# 2. Activation
# 3. Pooling
# 4. Fully connected layer

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(35,35,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,(3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
# Fully connected layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(4))
model.add(Activation('softmax'))


# In[195]:

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# In[33]:

gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

test_gen = ImageDataGenerator()


# In[36]:

train_generator = gen.flow(train_xx, train_yy)
test_generator = test_gen.flow(val_x, val_y)


# In[37]:

train_xx.shape


# In[38]:

val_x.shape


# In[203]:

model.fit_generator(train_generator, steps_per_epoch=7984//4, epochs=50, validation_data=test_generator, validation_steps=1996//4)


# In[204]:

score = model.evaluate(val_x, val_y, verbose=0)


# In[205]:

score


# In[181]:

test_x.shape


# In[182]:

pred = model.predict_proba(test_x)


# In[183]:

submit_df = pd.DataFrame(pred, columns=[1, 2, 3, 4])


# In[184]:

submit_df.head()


# In[185]:

submit_df.shape


# In[186]:

submit_df.to_csv('cnn_sub_15.csv')



