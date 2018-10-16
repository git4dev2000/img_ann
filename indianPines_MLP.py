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
class_weights = dict([(1,33),(2,200),(3,200),(4,181),(5,200),(6,200),(7,20),
                      (8,200),(9,14),(10,200),(11,200),(12,200),(13,143),(14,200),
                      (15,200),(16,75)])
patch_size = 1
lr = 1e-4
#units_1 = 200
units_2 = 2**8


# The corrected Salinas dataset has a coverage of 512*217(height*width) and 204 channels
# It represents 16 different classes. Each chnnel contains different ranges of values.
 
data_set = sio.loadmat(os.path.join(data_folder, data_file)).get('indian_pines_corrected')
gt = sio.loadmat(os.path.join(data_folder, gt_file)).get('indian_pines_gt')

# Preprocessing for data split for trainingand test
(train_rows, train_cols), (test_rows, test_cols) = preprocessing.data_split(gt, 
train_fraction=train_fraction, rem_classes=rem_classes, split_method=class_weights) # may set to 'same_hist'

# Reducing dataset dimension using PCA
#data_set = preprocessing.reduce_dim(data_set, .999)

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
nn_model.add(layer=layers.Dense(units=int(2/3*(num_catg+train_input.shape[3])), activation='relu'))
#nn_model.add(layer=layers.Dense(units=2**10, activation='relu'))
#nn_model.add(layer=layers.Dense(units=2**10, activation='relu'))
#nn_model.add(layer=layers.Dropout(0.35))
nn_model.add(layer=layers.Dense(units=num_catg, activation='softmax'))


# Compiling the modele
nn_model.compile(optimizer=optimizers.RMSprop(lr=lr),
                 loss=losses.categorical_crossentropy,
                 metrics=[metrics.categorical_accuracy])

history = nn_model.fit(x=train_input, y=y_train, batch_size=2**3, epochs=50,
                       validation_split=0.05)
    


# Plotting history
epoches = np.arange(1,len(history.history.get('loss'))+1)
plt.plot(epoches, history.history.get('loss'), 'b',label='Loss')
plt.plot(epoches, history.history.get('val_loss'),'bo', label='Validation_Loss')
plt.legend()    

plt.plot(epoches, history.history.get('categorical_accuracy'), 'b',label='Accuracy')
plt.plot(epoches, history.history.get('val_categorical_accuracy'),'bo', label='Validation_Accu')
plt.legend()

# Calculating metrics on test data
#
# Creating one_hot_2_int dictionary...
vectot_2_label = preprocessing.one_hot_2_label(int_to_vector_dict)
test_catgs, test_catg_counts = np.unique([vectot_2_label.get(tuple(elm)) for elm
                                          in y_test], return_counts=True)

# Generating a list of tuples for storing pixel coordinate, i.e
# with format (catg_lable, row, col, input_tensor, target_tensor, metric_container)
from_to_list = []
num_metrics = len(nn_model.metrics_names)
res_container = [(elm,[],[],[],[],[]) for elm in test_catgs]

i=0
for elm in test_catg_counts:
    from_idx = i 
    to_idx = i + elm
    i+=elm
    from_to_list.append((from_idx, to_idx))


for elm in zip(res_container, from_to_list): 
    elm[0][1].append(test_rows[elm[1][0]:elm[1][1]]) # catg row
    elm[0][2].append(test_cols[elm[1][0]:elm[1][1]]) # catg col
    x=test_input[elm[1][0]:elm[1][1], :, :, :]
    y=y_test[elm[1][0]:elm[1][1],:]
    #elm[0][3].append(test_input[elm[1][0]:elm[1][1], :, :, :]) # input_tensor
    #elm[0][4].append(y_test[elm[1][0]:elm[1][1],:]) # predicted tensor
    test_metrics= nn_model.evaluate(x=x, y=y) 
    elm[0][5].append(test_metrics) #metric

for elm in res_container:
    print((elm[0],elm[-1]))








