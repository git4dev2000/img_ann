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


data_folder = '/home/mansour/imgANN/Datasets'
data_file= 'Indian_pines_corrected'
gt_file = 'Indian_pines_gt'
train_fraction = 0.85 # with respect to the entire data. 
val_fraction = 0.05 #with rescept to training set.
rem_classes = [0]
patch_size = 1
lr = 1e-4
units_1 = 2**8
drop_rate =0.35
batch_size = 2**3
#units_2 = 2**8
class_weights = dict([(1,33),(2,200),(3,200),(4,181),(5,200),(6,200),(7,20),
                      (8,200),(9,14),(10,200),(11,200),(12,200),(13,143),(14,200),
                      (15,200),(16,75)])


# Loading dataset 
data_set = sio.loadmat(os.path.join(data_folder, data_file)).get('indian_pines_corrected')
gt = sio.loadmat(os.path.join(data_folder, gt_file)).get('indian_pines_gt')

# Preprocessing for data split for training and test
(train_rows, train_cols), (test_rows, test_cols) = preprocessing.data_split(gt, 
train_fraction=train_fraction, rem_classes=rem_classes, split_method='same_hist') # may set to 'same_hist'

# Reducing dataset dimension using PCA
#data_set = preprocessing.reduce_dim(data_set, .999)

# Scaling the input data using mean and std at each band
data_set = preprocessing.rescale_data(data_set)

# Preparing input tensors for training, validation and test sets:
(train_rows_sub, train_cols_sub), (val_rows, val_cols) = preprocessing.val_split(
        train_rows, train_cols, gt, val_fraction=val_fraction)

train_pixel_indices_sub = (train_rows_sub, train_cols_sub)
val_pixel_indices = (val_rows, val_cols)
test_pixel_indices = (test_rows, test_cols) 
catg_labels = np.unique([int(gt[idx[0],idx[1]]) for idx in zip(train_rows, train_cols)])
num_catg = len(catg_labels)
int_to_vector_dict = preprocessing.label_2_one_hot(catg_labels)

train_input_sub, y_train_sub = preprocessing.create_patch(
        data_set=data_set,
        gt=gt,
        pixel_indices=train_pixel_indices_sub,
        patch_size=patch_size,
        label_vect_dict=int_to_vector_dict)
val_input, y_val = preprocessing.create_patch(
        data_set=data_set,
        gt=gt,
        pixel_indices=val_pixel_indices,
        patch_size=patch_size,
        label_vect_dict=int_to_vector_dict)
test_input, y_test = preprocessing.create_patch(
        data_set=data_set,
        gt=gt,
        pixel_indices=test_pixel_indices,
        patch_size=patch_size,
        label_vect_dict=int_to_vector_dict)

# Building a MLP network model
input_shape = (patch_size, patch_size, data_set.shape[-1])
nn_model = models.Sequential()
#
# dense_input
nn_model.add(layer=layers.Dense(units=data_set.shape[2], activation='relu',
                                input_shape=input_shape))
# flatten_1, changes input shape from (1,1,num_band) to (num_band,)
nn_model.add(layer=layers.Flatten())
# dense_1
nn_model.add(layer=layers.Dense(units=units_1, activation='relu')) # could be set to units=int(2/3*(num_catg+input_shape[-1]))
# dropout_1
nn_model.add(layer=layers.Dropout(drop_rate))
# dense_output
nn_model.add(layer=layers.Dense(units=num_catg, activation='softmax'))


# Compiling the modele
nn_model.compile(optimizer=optimizers.RMSprop(lr=lr),
                 loss=losses.categorical_crossentropy,
                 metrics=[metrics.categorical_accuracy])

# Training the model
history = nn_model.fit(x=train_input_sub, y=y_train_sub, batch_size=batch_size,
                       epochs=50, validation_data=(val_input, y_val))
    

# Plotting history
epoches = np.arange(1,len(history.history.get('loss'))+1)
plt.plot(epoches, history.history.get('loss'), 'b',label='Loss')
plt.plot(epoches, history.history.get('val_loss'),'bo', label='Validation_Loss')
plt.legend()    

plt.plot(epoches, history.history.get('categorical_accuracy'), 'b',label='Accuracy')
plt.plot(epoches, history.history.get('val_categorical_accuracy'),'bo', label='Validation_Accu')
plt.legend()

#test_pixel_indices = (test_rows, test_cols)
model_metrics = preprocessing.calc_metrics(nn_model, test_input,
                                           y_test, int_to_vector_dict)














# Calculating metrics on test data
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
    #elm[0][1].append(test_rows[elm[1][0]:elm[1][1]]) # catg row
    #elm[0][2].append(test_cols[elm[1][0]:elm[1][1]]) # catg col
    x=test_input[elm[1][0]:elm[1][1], :, :, :]
    y=y_test[elm[1][0]:elm[1][1],:]
    #elm[0][3].append(test_input[elm[1][0]:elm[1][1], :, :, :]) # input_tensor
    #elm[0][4].append(y_test[elm[1][0]:elm[1][1],:]) # predicted tensor
    test_metrics= nn_model.evaluate(x=x, y=y) 
    elm[0][5].append(test_metrics) #metric

for elm in res_container:
    print((elm[0],elm[-1]))

#nn_model.save('indian_pines_MLP2_2.h5')

# Plotting predicted results
concat_rows =  np.concatenate((train_rows_sub, val_rows, test_rows))
concat_cols = np.concatenate((train_cols_sub, val_cols, test_cols))
concat_input = np.concatenate((train_input_sub, val_input, test_input))
concat_y = np.concatenate((y_train_sub, y_val, y_test))

y_pred_vectors = nn_model.predict(concat_input, batch_size=1)   
y_pred = np.zeros(y_pred_vectors.shape[0], dtype=int)
for elm in enumerate(y_pred_vectors):
    max_idx, *cols = np.where(elm[1]==np.amax(elm[1]))
    predicted_vec = np.eye(1,y_pred_vectors.shape[1],k=max_idx[0], dtype=int).ravel()
    y_pred[elm[0]] = vectot_2_label.get(tuple(predicted_vec))

gt_pred_map = np.zeros(gt.shape, dtype=int)
for elm in enumerate(zip(concat_rows, concat_cols)):
    gt_pred_map[elm[1]] = y_pred[elm[0]] 

plt.imshow(gt_pred_map)
##############################################################################
# Plotting prediction results for all pixels
rr, cc = np.meshgrid(np.arange(gt.shape[0]), np.arange(gt.shape[1]))
all_pixel_indices = (rr.ravel(), cc.ravel())
all_inputs, all_labels = preprocessing.create_patch(data_set,
                                                    gt,
                                                    all_pixel_indices,
                                                    patch_size=patch_size)

all_y_pred_vectors = nn_model.predict(all_inputs, batch_size=1)
all_y_pred=np.zeros(all_y_pred_vectors.shape[0], dtype=int)
for elm in enumerate(all_y_pred_vectors):
    max_idx, *cols = np.where(elm[1]==np.amax(elm[1]))
    predicted_vec = np.eye(1,all_y_pred_vectors.shape[1],k=max_idx[0], dtype=int).ravel()
    all_y_pred[elm[0]] = vectot_2_label.get(tuple(predicted_vec))

gt_pred_all_map = np.zeros(gt.shape, dtype=int)
for elm in enumerate(zip(rr.ravel(), cc.ravel())):
    gt_pred_all_map[elm[1]] = all_y_pred[elm[0]]

plt.imshow(gt_pred_all_map)
















