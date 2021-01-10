import warnings
warnings.filterwarnings('ignore')

import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

from glob import glob

import tensorflow as tf
import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor

from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Model

from keras import backend as K

from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from os import makedirs
from os.path import expanduser, exists, join

from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

from numpy import array, argmax

#%%
n_classes = 5
image_size = (299,299)
img_size = 299
batch_size = 32
n_epochs = 20
test_size = 0.2

#%%
# Get dataset from transformed hkl file (train .8, test .2)
import hickle as hkl

def load_hkl_dataset(inputfile):
    dset = hkl.load(inputfile)
    x_train = dset['x_train']
    x_validation = dset['x_validation']
    y_train = dset['y_train']
    y_validation = dset['y_validation']
    return [x_train, x_validation, y_train, y_validation]

x_train, x_validation, y_train, y_validation = load_hkl_dataset('dataset/thai-food_dset-8-2.hkl')

# Manually write label name in order of hkl file
source_enc = LabelEncoder()
source_enc.fit(['green curry', 'mango sticky rice', 'pad thai', 'thai papaya salad', 'tom yum'])

#%%
# Create train generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30, # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.3,
    width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
    horizontal_flip = 'true') # randomly flip images
train_generator = train_datagen.flow(x_train, y_train, shuffle=False, batch_size=batch_size, seed=10)

# Create validation generator
val_datagen = ImageDataGenerator(rescale = 1./255)
val_generator = val_datagen.flow(x_validation, y_validation, shuffle=False, batch_size=batch_size, seed=10)
#%%
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
    
augmented_images = [train_generator[0][0][0] for i in range(5)]
plotImages(augmented_images)

t_x, t_y = next(train_generator)
fig, m_axs = plt.subplots(2, 5, figsize = (16, 8))
for (tc_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_x = np.clip((tc_x-tc_x.min())/(tc_x.max()-tc_x.min())*255, 0 , 255).astype(np.uint8)[:,:,::]
    c_ax.imshow(c_x[:,:])
    c_ax.set_title('%s' % source_enc.classes_[np.argmax(c_y)])
    
t_x, t_y = next(val_generator)
fig, m_axs = plt.subplots(2, 5, figsize = (16, 8))
for (tc_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_x = np.clip((tc_x-tc_x.min())/(tc_x.max()-tc_x.min())*255, 0 , 255).astype(np.uint8)[:,:,::]
    c_ax.imshow(c_x[:,:])
    c_ax.set_title('%s' % source_enc.classes_[np.argmax(c_y)])    

#%%
# Get the InceptionV3 model so we can do transfer learning
base_model = InceptionV3(
    weights = 'imagenet', 
    include_top = False, 
    input_shape=(299, 299, 3))

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer and a logistic layer with 5 classes 
x = Dense(512, activation='relu')(x)
predictions = Dense(n_classes, activation='softmax')(x)

# The model we will train
model = Model(
    inputs = base_model.input, 
    outputs = predictions)

# train only the top layers i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# how many layers our model has
model.summary()
print(len(model.layers))
    
# Compile with Adam
model.compile(
    Adam(lr=.0001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy'])

# Save model every epoch
filepath = 'saved_pre-trained-model/saved-model-{epoch:02d}-{val_accuracy:.2f}.hdf5'
checkpoint = ModelCheckpoint(
    filepath,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=False,
    mode='auto',
    )
#%%
# Train the model
history = model.fit_generator(
    train_generator,
                      steps_per_epoch = int(len(x_train)/batch_size),
                      validation_data = val_generator,
                      validation_steps = int(len(x_validation)/batch_size),
                      epochs = n_epochs,
                      verbose = 2,
                      callbacks=[checkpoint])

#%%
# # Evaluation
# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss'] 

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc, label='training acc')
plt.plot(epochs, val_acc, label='validation acc')
plt.title('Training and validation accuracy of pretrained model')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss, label='training loss')
plt.plot(epochs, val_loss, label='validation lossS')
plt.title('Training and validation loss before of pretrained model')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()