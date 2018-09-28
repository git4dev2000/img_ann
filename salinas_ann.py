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
from matplotlib import pyplot as plt
import math
import preprocessing

# Loading dataset
data_folder = '/home/mansour/imgANN/Datasets'
data_file= 'Salinas_corrected'
gt_file = 'Salinas_gt'
rem_classes = [0,1]
# The corrected Salinas dataset has a coverage of 512*217(height*width) and 204 channels
# It represents 16 different classes. Each chnnel contains different ranges of values.
 
data_set = sio.loadmat(os.path.join(data_folder, data_file)).get('salinas_corrected')
gt = sio.loadmat(os.path.join(data_folder, gt_file)).get('salinas_gt')

# Preprocessing for data split for training, val, and test
(tr_rows2, tr_cols2), (te_rows2, te_cols2) = preprocessing.data_split(gt, rem_classes=[0,5,10,12])

# Image normilization and dimensionality reduction
u, s, v = np.linalg.svd(data_set[:,:,100])
max_dim = 30
u2=u[:,:max_dim]
s2=np.zeros((max_dim,217))
s2[:,:max_dim]=np.diag(s[:max_dim])
a_rec2 = np.dot(u2@s2,v)
plt.imshow(a_rec2)

# PCA test
import matplotlib.image as mpimg
img = mpimg.imread('wild.png')



     
