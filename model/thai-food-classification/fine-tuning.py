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

# IMPORTANT
# Import best pre-trained model manually !!
model_path = os.path.join('saved_pre-trained-model','saved-trained-model-07-89.hdf5')
model = keras.models.load_model(model_path)

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
# Fine-tuning
# set all layers trainable by default
for layer in model.layers:
    layer.trainable = True
    if isinstance(layer, BatchNormalization):
        # we do aggressive exponential smoothing of batch norm
        # parameters to faster adjust to our new dataset
        layer.momentum = 0.9
    
# fix deep layers (fine-tuning only last 50)
for layer in model.layers[:-50]:
    # fix all but batch norm layers, because we neeed to update moving averages for a new dataset!
    if not isinstance(layer, BatchNormalization):
        layer.trainable = False

for i, layer in enumerate(model.layers):
    print(i, layer.name)
#%%
from tensorflow.keras.optimizers import SGD

model.compile(
    optimizer=SGD(lr=0.00001, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )

filepath = 'saved_fine-tuned-model/saved-tuned-model-{epoch:02d}-{val_accuracy:.2f}.hdf5'
checkpoint = ModelCheckpoint(
    filepath,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=False,
    mode='auto',
    )

#%%
# Train the model
history = model.fit_generator(train_generator,
                      steps_per_epoch = int(len(x_train)/batch_size),
                      validation_data = val_generator,
                      validation_steps = int(len(x_validation)/batch_size),
                      epochs = n_epochs,
                      verbose = 2,
                      callbacks=[checkpoint])

#%%
# Evaluation
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
plt.title('Training and validation accuracy of fine-tuned model')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss, label='training loss')
plt.plot(epochs, val_loss, label='validation loss')
plt.title('Training and validation loss of fine-tuned model')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()