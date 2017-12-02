#-*- coding: utf-8 -*-
#import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import ipdb
import time
import math
import cv2
import codecs, json
from sklearn.metrics import average_precision_score
import pdb
import spacy
from random import seed
import itertools
from numpy import linalg as LA

#input_img_h5_train = './vqa_train_14by14_2.h5'
#input_img_h5_test = './spatial_data_img_residule_test_14by14.h5'
input_dir = './train_img_h5s_resize/'
output_dir = './train_img_h5s_resize_norm/'
def split_data():

    print('loading image feature...')
    with h5py.File(input_img_h5_train,'r') as hf:
        tem = hf.get('images_train')
        img_feature = np.array(tem).reshape(-1, 196, 2048)
        tem = LA.norm(img_feature, axis = 2)
        for i in range(tem.shape[0]):
            for j in range(tem.shape[1]):
                img_feature[i,j,:] = img_feature[i,j,:]/tem[i, j]
        for i in range(img_feature.shape[0]):
    	   output = output_dir + str(i+40000) + '.h5'
    	   print output
           img_f = img_feature[i,:,:]
    	   f = h5py.File(output, "w")
    	   f.create_dataset("img_f", dtype='float32', data=img_f)
    	   f.close()
    	   print i
    #current_img = img_feature_train[current_img_list,:]
def resize_data():
    print('resize image feature...')
    for i in range(82783):
        input = input_dir + str(i) + '.h5'
        output = output_dir + str(i) + '.h5'
        print input
        with h5py.File(input,'r') as hf:
           tem = hf.get('img_f')
           img_feature = np.array(tem).reshape(14,14,2048)
           print img_feature.shape
           img_f_1 = np.zeros((7,7,2048))
           for t in range(img_feature.shape[2]):
                img_f_1[:,:,t] = cv2.resize(img_feature[:,:,t],(0,0),fx=0.5, fy=0.5)
           print img_f_1.shape
           img_f = img_f_1.reshape(49,2048)
           f = h5py.File(output,"w")
           f.create_dataset("img_f",dtype='float32',data=img_f)
           f.close()
           print output
def norm_train():
    print('Norm the training data...')
    for i in range(82783):
        input = input_dir + str(i) + '.h5'
        output = output_dir + str(i) + '.h5'
        print input
        with h5py.File(input,'r') as hf:
           tem = hf.get('img_f')
           img_feature = np.array(tem).reshape(49, 2048)
           tem = LA.norm(img_feature, axis = 2)
           for i in range(tem.shape[0]):
                img_feature[i,:] = img_feature[i,:]/tem[i]
           f = h5py.File(output,"w")
           img_f = img_feature
           print img_f.shape
           f.create_dataset('img_f',dtype='float32',data=img_f)
           f.close()
           print output

if __name__ == '__main__':
#    split_data()
#    resize_data()
     norm_train()
