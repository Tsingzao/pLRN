#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:18:09 2017

@author: Tsingzao
"""

import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
os.system('echo $CUDA_VISIBLE_DEVICES')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import numpy as np
import h5py
np.random.seed(1337)  # for reproducibility

from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D
from keras.models import Model
import keras.backend as K

'''=======================Parameters========================================'''
RESOLUTION = 16
NUM_FRAME  = 16
NUM_CLASS  = 15
'''=======================TrainData========================================='''
BASIC_PATH = '/data/LowResolution/Penn_Train_'+str(RESOLUTION)+'.h5'
h5file     = h5py.File(BASIC_PATH,'r')
otrain_data= h5file['data'][:]
h5file.close()
data       = np.transpose(otrain_data, (0,1,2,4,3))
'''=======================TestData=========================================='''
BASIC_PATH = '/data/LowResolution/Penn_Test_'+str(RESOLUTION)+'.h5'
h5file     = h5py.File(BASIC_PATH,'r')
otest_data = h5file['data'][:]
h5file.close()
data_t     = np.transpose(otest_data, (0,1,2,4,3))
'''=======================AuxiliaryTrain===================================='''
PRO_PATH   = '/data/LowResolution/Penn_Train_'+str(RESOLUTION)+'_target.h5'
h5file     = h5py.File(PRO_PATH,'r')
ftrain_data= h5file['foreground'][:]
btrain_data= h5file['background'][:]
h5file.close()
ftrain_data= np.transpose(ftrain_data, (0,1,2,4,3))*100
btrain_data= np.transpose(btrain_data, (0,1,2,4,3))*100
'''=======================AuxiliaryTest====================================='''
PRO_PATH   = '/data/LowResolution/Penn_Test_'+str(RESOLUTION)+'_target.h5'
h5file     = h5py.File(PRO_PATH,'r')
ftest_data = h5file['foreground'][:]
btest_data = h5file['background'][:]
h5file.close()
ftest_data = np.transpose(ftest_data, (0,1,2,4,3))*100
btest_data = np.transpose(btest_data, (0,1,2,4,3))*100

'''=======================DefineNetArch====================================='''
def RankNet(input_shape):
    input_video = Input(input_shape)
    x = Conv3D(16,[3,3,3],padding='same',activation='relu', name='rank_conv1')(input_video)
    x = MaxPooling3D((2,2,2), name='rank_maxpool')(x)
    x = Conv3D(32,[3,3,3],padding='same',activation='relu', name='rank_conv2')(x)
    x = UpSampling3D(size=(2, 2, 2), name='rank_upsample')(x)
    x = Conv3D(16,[3,3,3],padding='same',activation='relu', name='rank_conv3')(x)
    y = Conv3D(1, [3,3,3],padding='same',activation='relu', name='rank_conv4')(x)
    return Model(input_video, y)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

model = RankNet((RESOLUTION, RESOLUTION, NUM_FRAME, 1))
model.summary()

model.compile(optimizer='sgd', loss=dice_coef_loss, metrics=[dice_coef])
#model.compile(optimizer='sgd', loss='mae')
model.fit(data, btrain_data, batch_size=4, epochs=100, verbose=1, validation_data=(data_t, btest_data))
