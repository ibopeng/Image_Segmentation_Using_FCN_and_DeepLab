# files include: current file and metrics.py
# input images with single channel. 
# Entries includes number: 0,1~20, 255
# return the result of comparing single prediction and its ground truth
import numpy as np
from matplotlib import pyplot as plt
from metrics import *
import os
import scipy.io as spio


def compare_prediction_and_label_of_segmentation(pred_image,label_image):
    # label_image: size, (height, width)
    # pred_image: size, (height, width)
    # height, width = label_image.shape
    
    # calculate pixel accuracy
    pix_acc = pixel_accuracy(pred_image, label_image)
    # calculate mean accuracy
    m_acc = mean_accuracy(pred_image, label_image)
    # calculate mean accuracy
    IoU = mean_IU(pred_image, label_image)
    # calculate weighted IOU accuracy
    freq_weighted_IU = frequency_weighted_IU(pred_image, label_image)
    
    return pix_acc, m_acc, IoU, freq_weighted_IU


dir_of_label = './label_npy/'
dir_of_prediction = './pred_npy/'
list_of_image_name = os.listdir(dir_of_label)
mean_IoU = []
mean_pixel_acc = []
mean_freq_weighted_IU = []
mean_acc = []
num_of_classes = 22
confusion_matrix = np.zeros((num_of_classes,num_of_classes))
for i in range(len(list_of_image_name)):
   
    # mat_name = list_of_image_name[i].split('.')[0] # get image name without suffix
    # pred_image = spio.loadmat('./prediciton_image/' + mat_name + '.mat')
    # label_image = spio.loadmat('./label_image/' + mat_name + '.mat')
    # here need convert .mat into numpy array with shape (height, width)
    # pred_image = pred_image['im']
    # label_image = label_image['im']
    
    # pred_image and label_image should have single channel with index of 0,1~20, 255
    npy_name = list_of_image_name[i] # get image name without suffix
    pred_image = np.load(dir_of_prediction + npy_name)  # numpy array with shape (height, width)
    label_image = np.load(dir_of_label + npy_name) # numpy array with shape (height, width)

    pix_acc, m_acc, IoU, freq_weighted_IU = compare_prediction_and_label_of_segmentation(pred_image, label_image)
    
    confusion_matrix += get_confusion_matrix(pred_image, label_image)
    
    mean_pixel_acc.append(pix_acc)
    mean_acc.append(m_acc)
    mean_IoU.append(IoU)
    mean_freq_weighted_IU.append(freq_weighted_IU)

print("Mean pixel accuracy:", np.mean(mean_pixel_acc))
print("Mean accuraccy:", np.mean(mean_acc))
print("Mean IoU:", np.mean(mean_IoU))
print("Mean frequency weighted IU:", np.mean(mean_freq_weighted_IU))
np.save('confusion_matrix', confusion_matrix)

