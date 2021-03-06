# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 12:06:53 2018

@author: Ryan_ye

refer to https://github.com/SimJeg/FC-DenseNet/blob/master/layers.py
"""

from tensorflow.keras.layers import Activation,Conv2D,MaxPooling2D,UpSampling2D,Dense,BatchNormalization,Input,Reshape,multiply,add,Dropout,AveragePooling2D,GlobalAveragePooling2D,concatenate
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import Model														  
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
# from keras.engine import Layer,InputSpec


def BN_ReLU_Conv(inputs, n_filters,L2_C, filter_size=3, dropout_p=0.2):
    '''Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0)''' 

    l = BatchNormalization()(inputs)
    l = Activation('relu')(l)
    l = Conv2D(n_filters, filter_size, padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(L2_C))(l)
    if dropout_p != 0.0:
        l = Dropout(dropout_p)(l,training=True)
    return l

def TransitionDown(inputs, n_filters,L2_C, dropout_p=0.2):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """
    l = BN_ReLU_Conv(inputs, n_filters,L2_C, filter_size=1, dropout_p=dropout_p)
    l = MaxPooling2D((2,2))(l)
    return l

def TransitionUp(skip_connection, block_to_upsample, n_filters_keep,L2_C):
    '''Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection'''
    #Upsample and concatenate with skip connection
    l = Conv2DTranspose(n_filters_keep, kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(L2_C))(block_to_upsample)
    l = concatenate([l, skip_connection], axis=-1)
    return l

def SoftmaxLayer(inputs, n_classes,L2_C):
    """
    Performs 1x1 convolution followed by softmax nonlinearity
    The output will have the shape (batch_size  * n_rows * n_cols, n_classes)
    """
    l = Conv2D(n_classes, kernel_size=1, padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(L2_C))(inputs)
    l = Reshape((-1, n_classes))(l)                               #### Uncommented by omid
    l = Activation('softmax')(l)#or softmax for multi-class
    return l
    
    