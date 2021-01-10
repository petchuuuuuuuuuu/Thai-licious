import warnings
warnings.filterwarnings('ignore')

import numpy as np  
import matplotlib.pyplot as plt
import os

import tensorflow as tf
import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from keras.models import Model

from sklearn.preprocessing import LabelEncoder

#%%
n_classes = 5
image_size = (299,299)
img_size = 299
batch_size = 32
n_epochs = 20
test_size = 0.2

# IMPORTANT
# Import model manually !!
model_path = os.path.join('saved_fine-tuned-model','saved-tuned-model-20-0.94.hdf5')
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
# Create validation generator
val_datagen = ImageDataGenerator(rescale = 1./255)
val_generator = val_datagen.flow(x_validation, y_validation, shuffle=False, batch_size=batch_size, seed=10)

#%%
from tensorflow.keras.optimizers import SGD

model.compile(
    optimizer=SGD(lr=0.00001, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )

#%%
# Evaluation from imported model
result = model.evaluate(val_generator)
dict(zip(model.metrics_names, result))

#%%
from sklearn.metrics import accuracy_score, classification_report

pred_Y = model.predict(val_generator, batch_size = None, verbose = True)
pred_Y_cat = np.argmax(pred_Y, -1)
test_Y_cat = np.argmax(y_validation, -1)
print('Accuracy on Test Data: %2.2f%%' % (100*accuracy_score(test_Y_cat, pred_Y_cat)))
print(classification_report(test_Y_cat, pred_Y_cat, target_names = source_enc.classes_))

#%%
category_names = source_enc.classes_

import seaborn as sns
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(test_Y_cat, pred_Y_cat)
fig, ax1 = plt.subplots(1,1, figsize = (5, 5))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cbar = True,
    cmap = plt.cm.Blues,
    vmax = x_train.shape[0]//16,
    ax = ax1,
    xticklabels=category_names,
    yticklabels=category_names,
    square=True
    )
plt.title('Confusion matrix of model') # title with fontsize 20
plt.xlabel('Predicted label') # x-axis label with fontsize 15
plt.ylabel('True label') # y-axis label with fontsize 15

plt.show()

#%%
# Visualizing Results
sample_imgs = 20
total_imgs = x_validation[0:sample_imgs].shape[0]
im_data = x_validation[0:sample_imgs]
im_label = test_Y_cat[0:sample_imgs]
label_names = source_enc.classes_[im_label]
pred_label = pred_Y_cat[0:sample_imgs]
pred_names = source_enc.classes_[pred_label]

fig, m_ax = plt.subplots(4, 5, figsize = (20, 20))
for c_ax, c_label, c_pred, c_img in zip(m_ax.flatten(), label_names, pred_names, im_data):
    tc_xs = c_img
    c_xs = np.clip((tc_xs-tc_xs.min())/(tc_xs.max()-tc_xs.min())*255, 0 , 255).astype(np.uint8)[:,:,::]
    c_ax.imshow(c_xs[:,:])
    c_ax.axis('off')
    c_ax.set_title('Predicted:{}\nActual:{}'.format(c_pred, c_label))