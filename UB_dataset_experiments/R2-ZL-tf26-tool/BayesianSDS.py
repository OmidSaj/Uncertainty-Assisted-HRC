# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:09:38 2019

@author: OMID
"""

from __future__ import absolute_import
from __future__ import print_function
import os
import numpy as np


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dropout,ZeroPadding2D,Cropping2D
from tensorflow.keras.layers import Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

from tensorflow.keras import backend as K


def BayesianSDS_buildModel(n_classes,input_shape,N_block,N_FL0,s_filter,drop_rate,N_blockdrop):
    xInput = Input(input_shape)
    # Encoder Blocks: 1- (N_block-1)
    for i in range(N_block-1):
        n_filter_i=int(2**(N_FL0+i))
        if i==0:
            x=Conv2D(n_filter_i, (s_filter,s_filter), padding="same", kernel_initializer='he_uniform')(xInput)
        else:
            x=Conv2D(n_filter_i, (s_filter,s_filter), padding="same", kernel_initializer='he_uniform')(x)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        x=MaxPooling2D(pool_size=(2, 2),padding="same")(x)
        
        if i+1>N_block-N_blockdrop:
            x=Dropout(drop_rate)(x,training=True)     # add dropout layer (Blocks before the last encoder)
    # Encoder Blocks: N_block
    n_filter_last=int(2**(N_FL0+N_block-1))
    x=Conv2D(n_filter_last, (s_filter,s_filter), padding="same", kernel_initializer='he_uniform')(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Dropout(drop_rate)(x,training=True)     # add dropout layer (Last encoder)
    # Decoder Blocks: 1
    x=Conv2D(n_filter_last, (s_filter,s_filter), padding="same", kernel_initializer='he_uniform')(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Dropout(drop_rate)(x,training=True)     # add dropout layer (First decoder)  
    # Decoder Blocks: 2-N_block
    for i in range(N_block-1):
        n_filter_i=int(2**(N_block-2+N_FL0-i))
        x=UpSampling2D(size=(2, 2))(x)
        x=Conv2D(n_filter_i, (s_filter,s_filter), padding="same", kernel_initializer='he_uniform')(x)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        
        if i<=N_block-N_blockdrop:
            x=Dropout(drop_rate)(x,training=True)     # add dropout layer (Blocks after the first decoder)  
    # Final Convolution Block    
    x=Conv2D(n_classes, (1, 1), padding="valid", kernel_initializer='he_uniform')(x)
    x=Reshape((input_shape[0]*input_shape[1], n_classes))(x)  
    y=Activation("softmax")(x)
    
    BaySDS_model=Model(inputs=xInput, outputs=y)
    # BaySDS_model.summary()
    return BaySDS_model

def BayesianSDS_buildModel_DbC(n_classes,input_shape,N_block,N_FL0,s_filter,drop_rate,N_blockdrop):
    xInput = Input(input_shape)
    # Encoder Blocks: 1- (N_block-1)
    for i in range(N_block-1):
        n_filter_i=int(2**(N_FL0+i))
        if i==0:
            x=Conv2D(n_filter_i, (s_filter,s_filter), padding="same")(xInput)
            x=Conv2D(n_filter_i, (s_filter,s_filter), padding="same")(x)
        else:
            x=Conv2D(n_filter_i, (s_filter,s_filter), padding="same")(x)
            x=Conv2D(n_filter_i, (s_filter,s_filter), padding="same")(x)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        x=MaxPooling2D(pool_size=(2, 2),padding="same")(x)
        
        if i+1>N_block-N_blockdrop:
            x=Dropout(drop_rate)(x,training=True)     # add dropout layer (Blocks before the last encoder)
    # Encoder Blocks: N_block
    n_filter_last=int(2**(N_FL0+N_block-1))
    x=Conv2D(n_filter_last, (s_filter,s_filter), padding="same")(x)
    x=Conv2D(n_filter_last, (s_filter,s_filter), padding="same")(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Dropout(drop_rate)(x,training=True)     # add dropout layer (Last encoder)
    # Decoder Blocks: 1
    x=Conv2D(n_filter_last, (s_filter,s_filter), padding="same")(x)
    x=Conv2D(n_filter_last, (s_filter,s_filter), padding="same")(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Dropout(drop_rate)(x,training=True)     # add dropout layer (First decoder)  
    # Decoder Blocks: 2-N_block
    for i in range(N_block-1):
        n_filter_i=int(2**(N_block-2+N_FL0-i))
        x=UpSampling2D(size=(2, 2))(x)
        x=Conv2D(n_filter_i, (s_filter,s_filter), padding="same")(x)
        x=Conv2D(n_filter_i, (s_filter,s_filter), padding="same")(x)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        
        if i<=N_block-N_blockdrop:
            x=Dropout(drop_rate)(x,training=True)     # add dropout layer (Blocks after the first decoder)  
    # Final Convolution Block    
    x=Conv2D(n_classes, (1, 1), padding="valid")(x)
    x=Reshape((input_shape[0]*input_shape[1], n_classes))(x)  
    y=Activation("softmax")(x)
    
    BaySDS_model=Model(inputs=xInput, outputs=y)
    BaySDS_model.summary()
    return BaySDS_model
# =============================================================================
# Bayesian SDS with custom loss function
# =============================================================================

#f = K.function([model.layers[0].input, K.learning_phase()],
#               [model.layers[-1].output])
#
#def predict_with_uncertainty(f, x, n_iter=10):
#    result = np.zeros((n_iter,) + x.shape)
#
#    for iter in range(n_iter):
#        result[iter] = f(x, 1)
#
#    prediction = result.mean(axis=0)
#    uncertainty = result.var(axis=0)
#    return prediction, uncertainty

from MonteCarloSettings import *

def custom_loss_Bayesian(y_true, y_pred):
    
    train_loss = K.categorical_crossentropy(y_true, y_pred)                    

    output_list = []
    for i in range(n_MoteCarlo_Samples):                                       
        output_list.append(K.categorical_crossentropy(y_true, y_pred))
    
    Monty_sample_bin = K.stack(output_list,axis=0)  
    val_loss_Bayesian=K.mean(Monty_sample_bin,axis=0)
    
    return K.in_train_phase(train_loss, val_loss_Bayesian)

# =============================================================================
# Updatable plot from: https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e
# =============================================================================
# updatable plot
# a minimal example (sort of)
from IPython.display import clear_output
import keras

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.accus=[]
        self.val_losses = []
        self.val_accus=[]
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.accus.append(logs.get('acc'))
        self.val_accus.append(logs.get('val_acc'))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        
        fig,ax=plt.subplots(1,2,figsize=(10,5))
        plt.subplots_adjust(wspace=0.2, hspace=None)
        
        ax[0].plot(self.x, self.losses,'b', label="loss")
        ax[0].plot(self.x, self.val_losses,'r', label="val_loss")
        ax[0].legend()

        ax[1].plot(self.x, self.accus,'b', label="training accuracy")
        ax[1].plot(self.x, self.val_accus,'r', label="validation accuracy")
        # ax[1].set_ylim(0.97,1)
        ax[1].legend()        
        
        tr_min_indx=np.argmin(self.losses)
        tr_min=min(self.losses)
        val_min_indx=np.argmin(self.val_losses)
        val_min=min(self.val_losses)
        
        ax[0].plot(tr_min_indx,tr_min,'xb')
        ax[0].plot(val_min_indx,val_min,'xr')
        
        clear_output(wait=True)
        

        
        plt.show();
        
plot_losses = PlotLosses()

#def val_loss_monitor(y_true, y_pred):
#    
#    val_loss_MC_Bin = K.zeros((n_MoteCarlo_Samples,) + (y_true.shape[0]) )   
#    for i in range(n_MoteCarlo_Samples):
#        val_loss_MC_Bin[i,:] = K.categorical_crossentropy(y_true, y_pred)
#    val_loss_Bayesian=K.mean(val_loss_MC_Bin,axis=0)
#    
#    return val_loss_B
#
#def mean_pred(y_true, y_pred):
#    return 100* K.categorical_crossentropy(y_true, y_pred)

def BaySDS_fit(model,X_train,Y_train,X_val,Y_val,verbose,
               class_weights_mat_train,class_weights_mat_val,
                            n_epoch,batch_size,model_filename_save,
                            patience,optimizer,hyp_a,hyp_b,hyp_c,hyp_d):

    if optimizer=='SGD':
        opt_method=optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    elif optimizer=='RMSprop':
        opt_method=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    elif optimizer=='Adagrad':
        opt_method=optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    elif optimizer=='Adadelta':
        opt_method=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    elif optimizer=='Adam':
        opt_method=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif optimizer=='Adamax':
        opt_method=optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    elif optimizer=='Nadam':
        opt_method=optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)      
    else:
        print('Error! optimizer not found!')    
    
    def step_decay(epoch):
        # initial_lrate = 1.0 # no longer needed
        lrate = hyp_a * np.exp(-hyp_b*epoch)
        return lrate

    if optimizer=='SGD':
        opt_method=optimizers.SGD(lr=0, momentum=0.8, decay=0, nesterov=False)
    elif optimizer=='RMSprop':
        opt_method=optimizers.RMSprop(lr=0, rho=0.9, epsilon=None, decay=0.0)
    elif optimizer=='Adagrad':
        opt_method=optimizers.Adagrad(lr=0, epsilon=None, decay=0)
    elif optimizer=='Adadelta':
        opt_method=optimizers.Adadelta(lr=0, rho=0.95, epsilon=None, decay=0.0)
    elif optimizer=='Adam':
        opt_method=optimizers.Adam(lr=0, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
    elif optimizer=='Adamax':
        opt_method=optimizers.Adamax(lr=0, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0)
    elif optimizer=='Nadam':
        opt_method=optimizers.Nadam(lr=0, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0)
    else:
        print('Error! optimizer not found!')
        
    lrate = LearningRateScheduler(step_decay)
    
    model.compile(loss=custom_loss_Bayesian, optimizer=opt_method, metrics=["accuracy"],sample_weight_mode="temporal")
    
    model_callbacks = [EarlyStopping(monitor='val_loss', patience=patience),
             ModelCheckpoint(filepath=model_filename_save, monitor='val_acc', save_best_only=True,mode='max'),lrate,plot_losses]  

    model_hist = model.fit(X_train, Y_train, batch_size=batch_size,epochs=n_epoch,verbose=verbose,
                           callbacks=model_callbacks,
                                  validation_data=(X_val,Y_val,class_weights_mat_val), sample_weight=class_weights_mat_train)
#    loss_history = SegDC_model_hist.history["loss"]
#    numpy_loss_history = np.array(loss_history)
#    np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")
    return model_hist,model
