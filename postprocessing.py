#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 12:37:31 2018

@author: mansour
"""
import numpy as np
from matplotlib import pyplot as plt


def calc_metrics(nn_model, test_inputs, y_test, int_to_vector_dict, verbose=True):
    """
    Calculates model performance metrics on test data.
    
    Arguments
    ---------
    nn_model : keras model.
        Trained neural network model containing metrics information.
    
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
    vector_2_label = one_hot_2_label(int_to_vector_dict)
    test_catgs, test_catg_counts = np.unique([vector_2_label.get(tuple(elm)) for elm
                                              in y_test], return_counts=True)
    
    # Generating a list of tuples for storing pixel coordinate, i.e
    # with format (catg_label, metric_container)
    from_to_list = []
    res_container = [(elm, []) for elm in test_catgs]
    
    i=0
    for elm in test_catg_counts:
        from_idx = i 
        to_idx = i + elm
        i+=elm
        from_to_list.append((from_idx, to_idx))
    
    
    for elm in zip(res_container, from_to_list): 
        x=test_inputs[elm[1][0]:elm[1][1], :, :, :]
        y=y_test[elm[1][0]:elm[1][1],:]
        test_metrics= nn_model.evaluate(x=x, y=y) 
        elm[0][-1].append(test_metrics) #metric
    
    model_metrics = dict([(elm[0], elm[-1]) for elm in res_container])
    if verbose:
        for key, val in model_metrics.items():
            print(key, val)
    return model_metrics

def plot_partial_map(nn_model, gt, pixel_indices, input_tensor, targ_tensor,
             int_to_vector_dict, plo=True):
    """
    Plots prediction map using a trained model and inputs.
    
    Arguments
    ---------
    nn_model : A trained keras neural network model.
        Trained using input data. 
    
    gt : numpy.ndarray
        A 2-D numpy array containing int labels.
        
    pixel_indices : tuple of arrays
        A tuple of length two containing arrays of rows and columns of input
        pixels with format: (row_array, col_array)
        
    input_tensor : numpy.ndarray
        Contains input_tensor consistent with the nn_model inputs.
        
    targ_tensor : numpy.ndarray
        Target tensor, containing one_hot format of label data.
        
    int_to_vector_dict : dictionary
        Associates int labels to their corresponding one_hot format. Can be
        created using label_2_one_hot function.
        
    plo : logical
        If True, plots the map.
        
    Returns
    -------
    gt_pred_map : numpy.ndarray
        A 2-D numpy.ndarray, representing predicted labels.
    
    """
    rows, cols = pixel_indices[0], pixel_indices[1]
    vect_2_label_dict = one_hot_2_label(int_to_vector_dict)
    y_pred_vectors = nn_model.predict(input_tensor, batch_size=1)   
    y_pred = np.zeros(y_pred_vectors.shape[0], dtype=int)
    for elm in enumerate(y_pred_vectors):
        max_idx, *not_used = np.where(elm[1]==np.amax(elm[1]))
        predicted_vec = np.eye(1,y_pred_vectors.shape[1],k=max_idx[0], dtype=int).ravel()
        y_pred[elm[0]] = vect_2_label_dict.get(tuple(predicted_vec))
    
    map_shape=gt.shape
    gt_pred_map = np.zeros(map_shape, dtype=int)
    for elm in enumerate(zip(rows, cols)):
        gt_pred_map[elm[1]] = y_pred[elm[0]] 
    
    if plo:
        plt.imshow(gt_pred_map)
    return gt_pred_map

def plot_full_map(nn_model, data_set, gt, int_to_vector_dict, patch_size, plo=True):
    """
    Plots prediction map for the entire pixels.
    
    Arguments
    ---------
    nn_model : keras model.
        Trained using input data. 
    data_set : numpy.ndarray
        Contains image data with 'channel_last' format, i.e., (height, width, channels)
        
    int_to_vector_dict : dictionary
        Associates int labels to their corresponding one_hot format. Can be
        created using label_2_one_hot function.
        
    patch_size : int
        Represents patch size used for nn_model.
        
    plo : logical
        If True, plots the map.
        
    Returns
    -------
    gt_pred_all_map : numpy.ndarray
        A 2-D numpy.ndarray, representing predicted labels for all pixels.
    """
    rr, cc = np.meshgrid(np.arange(gt.shape[0]), np.arange(gt.shape[1]))
    all_pixel_indices = (rr.ravel(), cc.ravel())
    vector_2_label = one_hot_2_label(int_to_vector_dict)
    all_inputs, all_labels = create_patch(data_set, gt, all_pixel_indices,
                                          patch_size, int_to_vector_dict)
    all_y_pred_vectors = nn_model.predict(all_inputs, batch_size=1)
    all_y_pred=np.zeros(all_y_pred_vectors.shape[0], dtype=int)
    for elm in enumerate(all_y_pred_vectors):
        max_idx, *not_used = np.where(elm[1]==np.amax(elm[1]))
        predicted_vec = np.eye(1,all_y_pred_vectors.shape[1],k=max_idx[0], dtype=int).ravel()
        all_y_pred[elm[0]] = vector_2_label.get(tuple(predicted_vec))
    gt_pred_all_map = np.zeros(gt.shape, dtype=int)
    for elm in enumerate(zip(rr.ravel(), cc.ravel())):
        gt_pred_all_map[elm[1]] = all_y_pred[elm[0]]
    
    if plo:
        plt.imshow(gt_pred_all_map)
    return gt_pred_all_map