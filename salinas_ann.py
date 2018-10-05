#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 12:54:16 2018

@author: mansour
"""
import numpy as np
from scipy import io as sio
from keras import models, layers, optimizers, metrics, losses
import os
#from matplotlib import pyplot as plt
import math
import preprocessing
from matplotlib import pyplot as plt
# Loading dataset

data_folder = '/home/mansour/imgANN/Datasets'
data_file= 'Salinas_corrected'
gt_file = 'Salinas_gt'
rem_classes = [0]
# The corrected Salinas dataset has a coverage of 512*217(height*width) and 204 channels
# It represents 16 different classes. Each chnnel contains different ranges of values.
 
data_set = sio.loadmat(os.path.join(data_folder, data_file)).get('salinas_corrected')
gt = sio.loadmat(os.path.join(data_folder, gt_file)).get('salinas_gt')

# Preprocessing for data split for trainingand test
(tr_rows, tr_cols), (te_rows, te_cols) = preprocessing.data_split(gt)

# Reducing dataset dimension using PCA
data_set = preprocessing.reduce_dim(data_set, .999)

# Testing create_path function: 
pixel_indices = (tr_rows, tr_cols)
tensor = preprocessing.create_patch(data_set=data_set,
                                    pixel_indices=pixel_indices,
                                    patch_size=5)

#




    
    
























