#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 10:30:56 2018

@author: mansour
"""
import numpy as np
import math
from sklearn.decomposition import PCA

#from scipy import io as sio
#from keras import models, layers, optimizers, metrics, losses
#import os
#from matplotlib import pyplot as plt


# Preprocessing for data split for training, val, and test
def data_split(gt, train_fraction=0.7, rem_classes=None,
               split_method='same_hist'):
    """
    Outpus list of row and column indices for training and test sets.
    
    Arguments
    ---------
    gt : numpy array
        A 2-D Numpy array, containing integer values representing class ids.
        
    train_fraction : float 
        The ratio of training size to the entire dataset.
        
    rem_classes : None or array_like
        list of class ids (integers) not to be included in analysis, e.g., class
        ids that do not have any ground truth values.
        
    split_method : 'same_hist' or a dictionary
        The dictionaries keys represent class label and values represent number
        of elemnt to be used for training in each class.
              
    Returns
    -------
    out : 2-D tuple
        Containts lists of rows and column indices for training
        and test sets: (train_rows, train_cols), (test_rows, test_cols)
    """

    if rem_classes is None:
        rem_classes = []
    
    catgs, counts = np.unique(gt, return_counts=True)
    mask = np.isin(catgs, rem_classes, invert=True)
    catgs, counts = catgs[mask], counts[mask]
    # Counts the number of values after removing rem_classes:
    num_pixels = sum(np.isin(gt,rem_classes, invert=True).ravel())
    catg_ratios = counts/np.sum(counts) 
    num_sample_catgs = np.array([math.floor(elm) for elm in
                                 (catg_ratios * num_pixels)], dtype='int32')   
    all_catg_indices = [np.where(gt==catg) for catg in catgs]
    # A 2-D tuple with first element representing number of samples per catg
    # and the second element a 2-D tuple containing row and column indices in
    # the gt array.
    catg_with_indices = zip(num_sample_catgs, all_catg_indices, catgs)
    train_rows, train_cols, test_rows, test_cols = [], [], [], []
   
    #####if else goes here....
    for elm in catg_with_indices:
        all_indices_per_catg = np.arange(elm[0], dtype='int32')
        if split_method == 'same_hist':
            rand_train_indices = np.random.choice(all_indices_per_catg,
                                                  size=int(math.floor(elm[0]*train_fraction)),
                                                  replace=False)
            rand_test_indices = np.setdiff1d(ar1=all_indices_per_catg,
                                             ar2=rand_train_indices, assume_unique=True)
        elif isinstance(split_method, dict):
            rand_train_indices = np.random.choice(all_indices_per_catg,
                                                  size=split_method.get(elm[2]),
                                                  replace=False)
            rand_test_indices = np.setdiff1d(ar1=all_indices_per_catg,
                                             ar2=rand_train_indices, assume_unique=True)
        else:
            raise ValueError('Please select a valid option')
            
        
        train_rows.append(elm[1][0][rand_train_indices])
        train_cols.append(elm[1][1][rand_train_indices])
        test_rows.append(elm[1][0][rand_test_indices])
        test_cols.append(elm[1][1][rand_test_indices])
        
    # Function for flattening lists of sequences...
    def list_combiner(x, init_list=None):
        if init_list is None:
            init_list=[]
        for elm in x:
            for sub_elm in elm:
                init_list.append(sub_elm)
        return init_list    
    
    # Combining indices for different categories...
    train_rows, train_cols = [list_combiner(elm) for elm in (train_rows, train_cols)]
    test_rows, test_cols = [list_combiner(elm) for elm in (test_rows, test_cols)]       
    
    return (train_rows, train_cols), (test_rows, test_cols)
    

# Dimensionality reduction using Principal Component analysis (PCA)
def reduce_dim(img_data, n_components=0.95):
    """
    Reduces spectral dimension of image data using PCA.
    
    Arguments
    ---------
    img_data : 3-D numpy.ndarray
        Contains image data with shape: (height, width, band).
        
    n_components : float between 0 and 1 or and int.
        If float, represents the minimum fraction of variance, explained by
        n_components. If integer, represents the number of components.
        
    Returns
    -------
    out : 3-D numpy.ndarray
        Contains transformed data with shape (height, width, n_components).
        

    """
    
    # Unravelling each band's data
    img_shape = img_data.shape
    img_unravel = np.zeros(shape=(img_shape[0]*img_shape[1],img_shape[2]))
    for i in range(img_shape[2]):
        img_unravel[:,i] = np.ravel(img_data[:,:,i])
        
    
    pca = PCA(n_components=n_components)    
    unravel_transformed = pca.fit_transform(img_unravel)
    
    # Reshaping transformed data:
    n_col = unravel_transformed.shape[1]
    img_data_transformed = np.zeros(shape=(img_shape[0], img_shape[1], n_col))
    for i in np.arange(n_col):
        img_data_transformed[:,:,i] = np.reshape(
                unravel_transformed[:,i], newshape=(img_shape[0], img_shape[1]))
                            
        
    return img_data_transformed


# Border corrections
def create_patch(data_set, gt, pixel_indices, patch_size=5, label_vect_dict=None):
    """
    Creates input tensors.
    
    Arguments
    ---------
    data_set : A 3-D numpy.ndarray
       Contains image data with format: (height, width, bands).
       
    gt : A 2-D numpy.ndarray
        Contains integers, representing different categories.
        
    pixel_indices : A seuence of two sequences.
        Contains lists of integers, representing training pixel rows and columns.
        e.g., (train_rows, train_cols), where train_rows and train_cols are list
        of integers.
    
    patch_size : An odd integer
        Represents patch size.
    
    label_vect_dict : None or an int to vector dictionary
        Associates int labels to a one-hot vector.
    
    Returns
    -------
    input_tensor : numpy.ndarray
        Input tensor with format: (num_samples, patch_size, patch_size, bands).
        
    target_tensor : numpy.ndarray
        Target tensor with one_hot format.
    
    """
    rows = pixel_indices[0]
    cols = pixel_indices[1]
    
    if len(rows) != len(cols):
        raise ValueError("Unmatched number of rows and columns. The number of"
                         " rows is {}, but the number of columns is {}"
                         .format(len(rows), len(cols)))
                         
    max_row, max_col = (data_set.shape[0]-1), (data_set.shape[1]-1)
    sample_size = len(rows) 
    input_tensor = np.zeros(shape=(sample_size, patch_size, patch_size, data_set.shape[2]))
    catg_labels = []
    # Selecting a training pixel coordinate
    for idx in np.arange(sample_size):
        patch = np.zeros(shape=(patch_size, patch_size, data_set.shape[2]))
        patch_center = (rows[idx], cols[idx])
        patch_top_row = patch_center[0] - patch_size // 2
        patch_left_col = patch_center[1] - patch_size // 2
        top_lef_idx = (patch_top_row, patch_left_col)
        # Extracting class label:
        catg_labels.append(gt[rows[idx], cols[idx]])        
        for i in np.arange(patch_size):
            for j in np.arange(patch_size):
                patch_idx = (top_lef_idx[0] + i, top_lef_idx[1] + j)
                if (patch_idx[0] >= 0) and (patch_idx[0] <= max_row) \
                and (patch_idx[1]>= 0) and (patch_idx[1] <= max_col):
                    patch[i, j,:] = data_set[patch_idx[0], patch_idx[1], :]
        input_tensor[idx, :, :, :] = patch
        
    if label_vect_dict is None:
        label_vect_dict = label_2_one_hot(np.unique(gt))
   
    target_tensor = np.array([label_vect_dict.get(label) for label in catg_labels])
    return input_tensor, target_tensor

# Converting a list of int labels to one-hot foramt
def label_2_one_hot(label_list):
    """
    Creates a dictionary containing class labels and their one-hot vector.
    
    Arguments
    ---------
    label_list : A list of integers representing class labels.
    
    Returns
    -------
    one_hot_dict : dictionary
        A dictionary with class labels of type int as keys and their one-hot 
        vector representation as values.
    
    """
    catgs = np.unique(label_list)
    num_catgs = len(catgs)
    one_hot_dict = dict([(elm[1], np.eye(1, num_catgs, elm[0]).ravel()) \
                           for elm in enumerate(catgs)])
    return one_hot_dict

def one_hot_2_label(int_to_vector_dict):
    """
    Converts integer to one_hot dictionary to a one_hot to integer dictionary. 
    dictionary
    
    Arguments
    ---------
    one_hot_ndarray : A numpy.ndarray
        Contains one-hot format of class labels.
    
    Returns
    -------
    tuple_to_int_dict : dictionary
        keys are tuples with one-hot format and values are integer class labels.
    """
    tuple_to_int_dict = dict([(tuple(val),key) for key, val in int_to_vector_dict.items()])
    return tuple_to_int_dict
    
def val_split(rows, cols, gt, val_fraction=0.1, rem_classes=None,
              split_method='same_hist'):
    if rem_classes is None:
        rem_classes=[-1]
    
    gt_no_test = np.zeros(shape=gt.shape, dtype='int').reshape(gt.shape)-1 
    for elm in zip(rows,cols):
        gt_no_test[rows, cols] = gt[rows, cols]
        
    (train_rows, train_cols), (val_rows, val_cols) = data_split(
    gt_no_test,
    1-val_fraction,
    rem_classes,
    split_method)
    
    return (train_rows, train_cols), (val_rows, val_cols) 

def rescale_data(data_set, method='standard'):
    """
    Rescales image dataset using different methods.
    
    Arguments
    ---------
    data_set : 3-D numpy.ndarray
        Containts image data with format: (height, width, channels).
        
    methodod : str 
        Represents rescaling method. Can take one of: 'standard', 'zero_mean',
        or 'min_max_norm', 'mean_norm'.
        
    Returns: rescaled_data
    """
    if (not isinstance(data_set, np.ndarray)) or (len(data_set.shape) !=3):
        raise ValueError('data_set must be a 3-D numpy array!')
    
    rescale_data = np.zeros(data_set.shape)
    if method == 'standard':
        for i in np.arange(data_set.shape[-1]):
            channel = data_set[:,:,i]
            rescale_data[:,:,i] = (channel - np.mean(channel)) / np.std(channel)
    elif method == 'zero_mean':
        for i in np.arange(data_set.shape[-1]):
            channel = data_set[:,:,i]
            rescale_data[:,:,i] = channel - np.mean(channel)
    elif method == 'min_max_norm':
        for i in np.arange(data_set.shape[-1]):
            channel = data_set[:,:,i]
            rescale_data[:,:,i] = (channel - np.amin(channel)) / (np.amax(channel) \
                        - np.amin(channel))
    elif method == 'mean_norm':
        for i in np.arange(data_set.shape[-1]):
            channel = data_set[:,:,i]
            rescale_data[:,:,i] = (channel - np.mean(channel)) / (np.amax(channel) \
                        - np.amin(channel))
    else:
        raise ValueError('{} is not a valid method.'.format(method))
    
    return rescale_data

def calc_metrics(nn_model, test_inputs, y_test, int_to_vector_dict):
    """
    Calculates model performance metrics on test data.
    
    Arguments
    ---------
    nn_model : Trained neural network model with metrics information.
    
    test_inputs : numpy.ndarray
        Input tensor containing test inputs.
        
    y_test : numpy.ndarray
        Contains target test data with one_hot format.
        
    int_to_vector_dict : a int to vector dictionary
        Associates class int category labels to its corresponding one_hot format.
    
    Returns
    -------
    model_metrics : dictionary
        A dictionary with int keys representing category labels and list of
        model error and performance metrics as values. 
    """
    vectot_2_label = one_hot_2_label(int_to_vector_dict)
    test_catgs, test_catg_counts = np.unique([vectot_2_label.get(tuple(elm)) for elm
                                              in y_test], return_counts=True)
    
    # Generating a list of tuples for storing pixel coordinate, i.e
    # with format (catg_lable, row, col, input_tensor, target_tensor, metric_container)
    from_to_list = []
    #test_rows = pixel_indices[0]
    #test_cols = pixel_indices[1]
    #num_metrics = len(nn_model.metrics_names)
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
        x=test_inputs[elm[1][0]:elm[1][1], :, :, :]
        y=y_test[elm[1][0]:elm[1][1],:]
        #elm[0][3].append(test_input[elm[1][0]:elm[1][1], :, :, :]) # input_tensor
        #elm[0][4].append(y_test[elm[1][0]:elm[1][1],:]) # predicted tensor
        test_metrics= nn_model.evaluate(x=x, y=y) 
        elm[0][-1].append(test_metrics) #metric
    
    model_metrics = dict([(elm[0], elm[-1]) for elm in res_container])
    return model_metrics
    
   
    

    
    
    
    
    
    
    
    
    
    
