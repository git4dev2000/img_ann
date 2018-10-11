#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 12:54:16 2018

@author: mansour
"""
import numpy as np
from scipy import io as sio
from keras import models, layers, optimizers, metrics, losses, regularizers
import os
#from matplotlib import pyplot as plt
import math
import preprocessing
from matplotlib import pyplot as plt
# Loading dataset

data_folder = '/home/mansour/imgANN/Datasets'
data_file= 'Indian_pines_corrected'
gt_file = 'Indian_pines_gt'
train_fraction = 0.85
rem_classes = [0]
patch_size = 1
lr = 1e-3
#units_1 = 200
units_2 = 2**8





# The corrected Salinas dataset has a coverage of 512*217(height*width) and 204 channels
# It represents 16 different classes. Each chnnel contains different ranges of values.
 
data_set = sio.loadmat(os.path.join(data_folder, data_file)).get('indian_pines_corrected')
gt = sio.loadmat(os.path.join(data_folder, gt_file)).get('indian_pines_gt')

# Preprocessing for data split for trainingand test
(train_rows, train_cols), (test_rows, test_cols) = preprocessing.data_split(gt, 
train_fraction=train_fraction, rem_classes=rem_classes)

# Reducing dataset dimension using PCA
data_set = preprocessing.reduce_dim(data_set, .999)

# Scaling the input data using mean and std at each band
data_set_scaled = np.zeros(data_set.shape)
for i in np.arange(data_set.shape[2]):
    band = data_set[:,:,i]
    data_set_scaled[:,:,i] = (band - np.mean(band)) / np.std(band) # zero mean and 1 std
    #data_set_scaled[:,:,i] = band / np.amax(band)  # scaled with max
data_set = data_set_scaled
    
# Creating input tensor and class labels
train_pixel_indices = (train_rows, train_cols)
train_input, train_labels = preprocessing.create_patch(data_set=data_set,
                                    gt=gt,
                                    pixel_indices=train_pixel_indices,
                                    patch_size=patch_size)

test_pixel_indices = (test_rows, test_cols) 
test_input, test_labels = preprocessing.create_patch(data_set=data_set,
                                    gt=gt,
                                    pixel_indices=test_pixel_indices,
                                    patch_size=patch_size)



# Calculating number of categories
num_catg = len(np.unique(train_labels))

# Creating label to one-hot dictionary...
int_to_vector_dict = preprocessing.lebel_2_one_hot(train_labels)

# Creating label tensor to be used for a nn_model
y_train = np.array([int_to_vector_dict.get(elm) for elm in train_labels])
y_test = np.array([int_to_vector_dict.get(elm) for elm in test_labels])



# Building a MLP network model
nn_model = models.Sequential()
#
# dense_1
nn_model.add(layer=layers.Dense(units=data_set.shape[2], activation='relu',
                                input_shape=train_input.shape[1:]
                                ))
# flatten to chnage input shape from (1,1,num_band) to (num_band,)
nn_model.add(layer=layers.Flatten())
#nn_model.add(layer=layers.Dense(units=int(2/3*(num_catg+train_input.shape[3])), activation='relu'))
nn_model.add(layer=layers.Dense(units=2**10, activation='relu'))
#nn_model.add(layer=layers.Dropout(0.5))
nn_model.add(layer=layers.Dense(units=num_catg, activation='softmax'))


# Compiling the modele
nn_model.compile(optimizer=optimizers.RMSprop(lr=lr),
                 loss=losses.categorical_crossentropy,
                 metrics=[metrics.categorical_accuracy])

history = nn_model.fit(x=train_input, y=y_train, batch_size=2**5, epochs=30,
                       validation_split=0.1)
    

# Plotting history
epoches = np.arange(1,len(history.history.get('loss'))+1)
plt.plot(epoches, history.history.get('loss'), 'b',label='Loss')
plt.plot(epoches, history.history.get('val_loss'),'bo', label='Validation_Loss')
plt.legend()    

plt.plot(epoches, history.history.get('categorical_accuracy'), 'b',label='Accuracy')
plt.plot(epoches, history.history.get('val_categorical_accuracy'),'bo', label='Validation_Accu')
plt.legend()




















