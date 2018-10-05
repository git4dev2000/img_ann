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
def data_split(gt, train_fraction=0.7, rem_classes=None):
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
    catg_with_indices = zip(num_sample_catgs, all_catg_indices)
    train_rows, train_cols, test_rows, test_cols = [], [], [], []
    for elm in catg_with_indices:
        all_indices_per_catg = np.arange(elm[0], dtype='int32')
        rand_train_indices = np.random.choice(all_indices_per_catg,
                                              size=int(math.floor(elm[0]*train_fraction)),
                                              replace=False)
        rand_test_indices = np.setdiff1d(ar1=all_indices_per_catg,
                                         ar2=rand_train_indices, assume_unique=True)
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
def create_patch(data_set, pixel_indices, patch_size=5):
    """
    Creates input tensors.
    
    Arguments
    ---------
    data_set : A 3-D numpy.ndarray
       Contains image data with format: (height, width, bands).
       
    patch_size : An odd integer
        Represents patch size.
        
    pixel_indices : A seuence of two sequences.
        Contains lists of integers, representing training pixel rows and columns.
        e.g., (train_rows, train_cols), where train_rows and train_cols are list
        of integers.
    
    Returns
    -------
    out : Input tensor
        Input tensor with format: (num_samples, patch_size, patch_size, bands).
    """
    rows = pixel_indices[0]
    cols = pixel_indices[1]
    
    if len(rows) != len(cols):
        raise ValueError("Unmatched number of rows and columns. The number of"
                         " rows is {}, but the number of columns is {}"
                         .format(len(rows), len(cols)))
                         
    max_row, max_col = (data_set.shape[0]-1), (data_set.shape[1]-1)
    sample_size = len(rows) 
    patch_tensor = np.zeros(shape=(sample_size, patch_size, patch_size, data_set.shape[2]))
    # Selecting a training pixel coordinate
    for idx in np.arange(sample_size):
        patch = np.zeros(shape=(patch_size, patch_size, data_set.shape[2]))
        patch_center = (rows[idx], cols[idx])
        patch_top_row = patch_center[0] - patch_size // 2
        patch_left_col = patch_center[1] - patch_size // 2
        top_lef_idx = (patch_top_row, patch_left_col)
        
        for i in np.arange(patch_size):
            for j in np.arange(patch_size):
                patch_idx = (top_lef_idx[0] + i, top_lef_idx[1] + j)
                if (patch_idx[0] >= 0) and (patch_idx[0] <= max_row) \
                and (patch_idx[1]>= 0) and (patch_idx[1] <= max_col):
                    patch[i, j,:] = data_set[patch_idx[0], patch_idx[1], :]
        patch_tensor[idx, :, :, :] = patch      
        
    return patch_tensor
        
    


        
        
    
    

    
   
    

    
    
    
    
    
    
    
    
    
    
