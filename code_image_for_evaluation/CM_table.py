# -*- coding: utf-8 -*-
"""
Created on Sun May  6 14:36:30 2018

@author: Yiwu Zhong
"""

import numpy as np
DL20 = np.load('./deeplab_20/confusion_matrix.npy')
for j in range(DL20.shape[1]-1): # excludes the last col, namely class "255"
    col = np.sum(DL20[:,j])
    for i in range(DL20.shape[0]):
        DL20[i][j] = DL20[i][j] / col
        
DL60 = np.load('./deeplab_60/confusion_matrix.npy')
for j in range(DL60.shape[1]-1): # excludes the last col, namely class "255"
    col = np.sum(DL60[:,j])
    for i in range(DL60.shape[0]):
        DL60[i][j] = DL60[i][j] / col
        
DL100 = np.load('./deeplab_100/confusion_matrix.npy')
for j in range(DL100.shape[1]-1): # excludes the last col, namely class "255"
    col = np.sum(DL100[:,j])
    for i in range(DL100.shape[0]):
        DL100[i][j] = DL100[i][j] / col
        
FCN = np.load('./fcn/confusion_matrix.npy')
for j in range(FCN.shape[1]-1): # excludes the last col, namely class "255"
    col = np.sum(FCN[:,j])
    for i in range(FCN.shape[0]):
        FCN[i][j] = FCN[i][j] / col