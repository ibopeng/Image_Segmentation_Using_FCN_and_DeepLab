# -*- coding: utf-8 -*-
"""
Created on Sun May  6 20:12:27 2018

@author: Yiwu Zhong
"""
import numpy as np
import matplotlib.pyplot as plt
result = np.zeros((4,2))
result[0,0] = 0.8115
result[0,1] = 0.5625
result[1,0] = 0.8542
result[1,1] = 0.5935
result[2,0] = 0.8808
result[2,1] = 0.6423
result[3,0] = 0.8861
result[3,1] = 0.6579

dataset = np.array([[20], [60], [100]]) # size(3,1)
plt.scatter(dataset, result[1:4,0], 20, 'g') # [a,b)
plt.scatter(dataset, result[1:4,1], 20, 'b') # [a,b)
plt.plot(dataset, result[1:4,0], 'g', label='Mean Pixel Accuracy')
plt.plot(dataset, result[1:4,1],  'b', label='Mean IoU')
#plt.xlim(0, 1.1)
#plt.ylim(0, 1.1)
plt.grid()
plt.title('Learning Curve of Deeplab Models')
plt.xlabel('Training Data Size (%)')
plt.ylabel('Performance')
plt.legend()  # show the 'label' of line
plt.show()

name_list = ['Mean Pixel Accuracy','Mean IoU']  
FCN = result[0,:]
Deeplab_100_percent = result[3,:]  
x =list(range(len(FCN)))  # location of axis x  
n = len(name_list)  # number of objects which are compared
total_width = 1  # width of axis x 
width = total_width / 3   
  
plt.bar(x, FCN, width, label='FCN',fc = 'g')  

for i in range(len(x)):  
    x[i] = x[i] + width/2 + 0.005
plt.bar(x, Deeplab_100_percent, 0.01 ,tick_label = name_list,fc = 'w')  # center white interval

for i in range(len(x)):  
    x[i] = x[i] + width/2 + 0.005
plt.bar(x, Deeplab_100_percent, width, label='Deeplab',fc = 'b')  
plt.title('Comparison between FCN and Deeplab Models')
plt.xlabel('Metrics')
plt.ylabel('Performance')
plt.legend()
plt.show()  