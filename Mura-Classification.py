#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[30]:


import csv
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[31]:


import os
from glob import glob


# In[32]:


import keras
import keras.backend as k


# In[33]:


os.listdir()


# In[34]:



PATH="MURA-v1.1/"


# In[35]:



os.listdir(PATH)


# In[36]:


train_imgs_path=pd.read_csv(PATH+'train_image_paths.csv')
train_labels=pd.read_csv(PATH+'train_labeled_studies.csv')
test_imgs_path=pd.read_csv(PATH+'valid_image_paths.csv')
test_labels=pd.read_csv(PATH+'valid_labeled_studies.csv')


# In[37]:



train_imgs_path.head()


# In[38]:



train_imgs_path.shape


# In[44]:



train_labels.head(20)


# In[45]:



train_labels['1'].value_counts()


# In[46]:



test_imgs_path.head(30)


# In[47]:



test_imgs_path.shape


# In[ ]:



for path in train_imgs_path.values[:10]:
    img=cv2.imread(path[:10])
    plt.imshow(plt.imread(path[:10]))
    plt.imshow(img)
    plt.show()
    print (img.shape)


# In[55]:


train_labels['Body_Part']=train_labels['MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/'].apply(lambda x: str(x.split('/')[2])[3:])
train_labels['Study_Type']=train_labels['MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/'].apply(lambda x: str(x.split('/')[4])[:6])
test_labels['Body_Part']=test_labels['MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/'].apply(lambda x: str(x.split('/')[2])[3:])
test_labels['Study_Type']=test_labels['MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/'].apply(lambda x: str(x.split('/')[4])[:6])


# In[58]:



train_labels.head()


# In[59]:



test_labels.head()


# In[60]:



import seaborn as sns


# In[62]:



sns.countplot(train_labels['1'])


# In[63]:



plt.figure(figsize=(15,7))
sns.countplot(data=train_labels,x='Body_Part',hue='1')


# In[280]:


plt.figure(figsize=(15,7))
sns.countplot(data=test_labels,x='Body_Part',hue='1')


# In[64]:


plt.figure(figsize=(15,7))
sns.countplot(data=train_labels,x='Study_Type',hue='1')


# In[66]:



sns.countplot(test_labels['1'])


# In[67]:


plt.figure(figsize=(15,7))
sns.countplot(data=test_labels,x='Study_Type',hue='1')


# In[68]:


from tqdm import tqdm
from PIL import Image


# In[78]:


def read_image(Path):
    img=cv2.imread(Path)
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(224,224))
    #print (img.shape)
    img=np.array(img)
    #img=np.resize(img,(224,224))
    #print (img.shape)
    img=img/255.
    return img


# In[79]:


X_train=[]
X_val=[]


# In[82]:


import torch


# In[83]:


torch.cuda.is_available()


# In[88]:


train_labels['MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/']=train_imgs_path['MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/image1.png']
test_labels['MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/']=test_imgs_path['MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/image1.png']


# In[89]:


train_df=train_labels.groupby(['1']).apply(lambda x: x.sample(5000,replace=True))


# In[90]:



train_df.shape


# In[91]:



train_labels.shape


# In[92]:



sns.countplot(data=train_df,x='Body_Part',hue='1')


# In[109]:



from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions


# In[117]:



IMG_SIZE=(224,224)
def __init__(contrast_stretching=False,#####
histogram_equalization=False,#####
adaptive_equalization=False,#####
 data_format=None):
 if data_format is None:

   data_format = K.image_data_format()

   self.counter = 0

   self.contrast_stretching = contrast_stretching #####

   self.adaptive_equalization = adaptive_equalization #####

   self.histogram_equalization = histogram_equalization #####


# In[273]:


datagen=ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
    shear_range=0.4,
    zoom_range=0.4,
    fill_mode='nearest',
    preprocessing_function=preprocess_input,
)


# In[249]:


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


# In[250]:


train_gen = flow_from_dataframe(datagen, train_df, 
                             path_col = 'MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/',
                            y_col = '1', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 64)


# In[251]:



valid_gen = flow_from_dataframe(datagen, test_labels, 
                             path_col = 'MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/',
                            y_col = '1', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 64)


# In[252]:



test_X, test_Y = next(flow_from_dataframe(datagen, 
                               test_labels, 
                             path_col = 'MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/',
                            y_col = '1', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = 64)) # one big batch
# used a fixed dataset for final evaluation
final_test_X, final_test_Y = next(flow_from_dataframe(datagen, 
                            test_labels, 
                            path_col = 'MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/',
                            y_col = '1',
                            target_size = IMG_SIZE,
                            color_mode = 'rgb',
                            batch_size = 64)) # one big batch


# In[253]:



t_x,t_y=next(train_gen)


# In[254]:


fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -127, vmax = 127)
    c_ax.set_title('%s' % ('Pos' if c_y>0.5 else 'Neg'))
    c_ax.axis('off')


# In[255]:


from keras.layers import  Convolution2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.models import  Sequential


# In[279]:


model = Sequential()
model.add(Convolution2D(32, (3,3), activation='relu', padding='same',input_shape = t_x.shape[1:]))
#if you resize the image above, change the input shape
model.add(Convolution2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='softmax'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()


# In[259]:


base_model=VGG16(input_shape=t_x.shape[1:],include_top=False,weights='imagenet')
base_model.trainable=False


# In[260]:



base_model.summary()


# In[261]:


from keras.layers import *
from keras.models import *


# In[262]:


pt_features = Input(base_model.get_output_shape_at(0)[1:], name = 'feature_input')
pt_depth = base_model.get_output_shape_at(0)[-1]


# In[263]:


pt_features


# In[264]:



pt_depth


# In[275]:


bn_features = BatchNormalization()(pt_features)
# here we do an attention mechanism to turn pixels in the GAP on an off
attn_layer = Conv2D(128, kernel_size = (1,1), padding = 'same', activation = 'relu')(bn_features)
attn_layer = Conv2D(32, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
attn_layer = AvgPool2D((2,2), strides = (1,1), padding = 'same')(attn_layer) # smooth results
attn_layer = Conv2D(1, 
                    kernel_size = (1,1), 
                    padding = 'valid', 
                    activation = 'sigmoid')(attn_layer)
# fan it out to all of the channels
up_c2_w = np.ones((1, 1, 1, pt_depth))
up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 
               activation = 'linear', use_bias = False, weights = [up_c2_w])
up_c2.trainable = False
attn_layer = up_c2(attn_layer)

mask_features = multiply([attn_layer, bn_features])
gap_features = GlobalAveragePooling2D()(mask_features)
gap_mask = GlobalAveragePooling2D()(attn_layer)
# to account for missing values from the attention model
gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
gap_dr = Dropout(0.5)(gap)
dr_steps = Dropout(0.5)(Dense(128, activation = 'elu')(gap_dr))
out_layer = Dense(1, activation = 'sigmoid')(dr_steps)

attn_model = Model(inputs = [pt_features], outputs = [out_layer], name = 'attention_model')

attn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                           metrics = ['binary_accuracy'])

attn_model.summary()


# In[277]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('cardio_attn')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=10) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[278]:


from keras.optimizers import Adam
from keras.util.Sequence import Sequence


# In[268]:


model = Sequential(name = 'combined_model')
model.add(base_model)
model.add(attn_model)
model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy',
                           metrics = ['binary_accuracy'])
model.summary()


# In[272]:


train_gen.batch_size = 64
model.fit_generator(train_gen, 
                      validation_data = (test_X, test_Y), 
                    steps_per_epoch=100,
                      epochs = 3, 
                      callbacks = callbacks_list,
                      workers = 3)


# In[ ]:




