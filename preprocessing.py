#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 10:30:56 2018

@author: mansour
"""
import numpy as np
import math
#from scipy import io as sio
#from keras import models, layers, optimizers, metrics, losses
#import os
#from matplotlib import pyplot as plt


# Preprocessing for data split for training, val, and test
def data_split(gt, train_fraction=0.7, split_type='same_hist',
               rem_classes=None):
    """
    Outpus list of row and column indices for training and test sets.
    
    Arguments
    ---------
    gt : numpy array
        A 2-D Numpy array, containing integer values representing class ids.
        
    train_fraction : float 
        The ratio of training size to the entire dataset.
    
    split_type : str 
        Represents split method. Possible values are:
        'same_hist' (having the same class histogram for training, test and the
        entire datasets), 'random_split' (randomly selects the training and test
        dataset without preserving class histograms).
        
    rem_classes : None or array_like
        list of class ids (integers) not to be included in analysis, e.g., class
        ids that do not have any ground truth values. 
              
    Returns
    -------
    out : 2-D tuple
        Containts lists of rows and column indices for training
        and test sets: (train_rows, train_cols), (test_rows, test_cols)
    """
    
    catgs, counts = np.unique(gt, return_counts=True)
    num_train = math.floor(train_fraction * (gt.shape[0]*gt.shape[1]))    

    if rem_classes is not None:
        mask = np.isin(catgs, rem_classes, invert=True)
        catgs, counts = catgs[mask], counts[mask]
        # Counts the number of values after removing rem_classes:
        num_train = sum(np.isin(gt,rem_classes, invert=True).ravel())
    
    
    catg_ratios = counts/np.sum(counts) 
    num_train_catgs = np.array([math.floor(elm) for elm in (catg_ratios * num_train)], dtype='int32')   
    catg_indices = [np.where(gt==catg) for catg in catgs]
    train_rows=[]
    train_cols=[]
    test_rows=[]
    test_cols=[]
    for elm in enumerate(catg_indices):
        train_indices = np.random.choice(np.arange(len(elm[1][0])),
                                         size=num_train_catgs[elm[0]], replace=False)
        test_indices = np.setdiff1d(ar1=np.arange(len(elm[1][0])), ar2=train_indices)
        train_rows.append(elm[1][0][train_indices])
        train_cols.append(elm[1][1][train_indices])
        test_rows.append(elm[1][0][test_indices])
        test_cols.append(elm[1][1][test_indices])
        
    return (train_rows, train_cols), (test_rows, test_cols)


   

    
   
    

    
    
    
    
    
    
    
    
    
    
