# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:34:06 2019

@author: OMID
"""

from BayesianSDS import *

from keras import backend as K
from keras.utils import np_utils
import timeit
import datetime

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# One-hot encoding

def to_categorical(y, nb_classes):
    num_samples = len(y)
    Y = np_utils.to_categorical(y.flatten(), nb_classes)
    return Y.reshape((num_samples, int(y.size/num_samples), nb_classes))

# Import Data

# data_train = np.load(Main_dir+'Training.npz')
# X_train=np.array(data_train['image_bin'])
# Y_train=np.array(data_train['label_bin'])
# Y_train_cat=to_categorical(Y_train, n_classes)

# data_val = np.load(Main_dir+'Val.npz')
# X_val=np.array(data_val['image_bin'])
# Y_val=np.array(data_val['label_bin'])
# Y_val_cat=to_categorical(Y_val, n_classes)

# data_test = np.load(Main_dir+'Test.npz')
# X_test=np.array(data_test['image_bin'])
# Y_test=np.array(data_test['label_bin'])
# Y_test_cat=to_categorical(Y_test, n_classes)

# Import shuffled indecies and generate data splits (predefiend)
# filename='Hand-Shuffled-indx'
# dataset_x = np.load(Main_dir+filename+'.npz')

# train_indc=dataset_x['a']
# val_indc=dataset_x['b']
# test_indc=dataset_x['c']

# X_train=X_data[train_indc,:,:,:]
# X_val=X_data[val_indc,:,:,:]
# X_test=X_data[test_indc,:,:,:]

# Y_train=Y_data[train_indc,:,:]
# Y_val=Y_data[val_indc,:,:]
# Y_test=Y_data[test_indc,:,:]

# Y_train_cat=Y_cat[train_indc,:,:]
# Y_val_cat=Y_cat[val_indc,:,:]
# Y_test_cat=Y_cat[test_indc,:,:]

## Class weights manual selection
# pixel_unique,pixel_count=np.unique(Y_train,return_counts=True)
# # fig, axs = plt.subplots(figsize=(10, 5))
# pixel_median=np.median(pixel_count[1:]) 
# class_weights_freq=pixel_median/pixel_count   #np.arange(n_classes)      #pixel_median/pixel_count


# Function to generate keras sample weights
def gen_weight_mat(Y_data,CNN_weights):
    n_classes=CNN_weights.shape[0]    
    nobs=Y_data.shape[0]
    nnode_x=Y_data.shape[1]
    nnode_y=Y_data.shape[2]
    Target_reshaped=np.reshape(Y_data,(-1,nnode_x*nnode_y))

    class_weights_mat = np.zeros((nobs, nnode_x*nnode_y))
    for i in range(n_classes):    # imbalance weights
        class_weights_mat[Target_reshaped==i]=CNN_weights[i]    
        
    return class_weights_mat

# This usually required for evaluation and trianing 
# input_shape=X_train.shape[1:]


# # Nodal pixel counts for the ML decision rule
# nnode_along_y=X_test.shape[1]
# nnode_along_x=X_test.shape[2]
# pixel_count_node=np.zeros((nnode_along_y,nnode_along_x,2))
# for i in range(nnode_along_y): # rows
#     for j in range(nnode_along_x): # columns
#         pixel_unique,pixel_count_tmp=np.unique(Y_test[:,i,j],return_counts=True)
#         count=-1
#         for k in pixel_unique:
#             count+=1
#             k=k.astype(int)
#             pixel_count_node[i,j,k]=pixel_count_tmp[count]
# pixel_count_plot=pixel_count_node
# pixel_count_node=np.reshape(pixel_count_node,(-1,2))


# =============================================================================
# Bayesian Inference
# =============================================================================
from MonteCarloSettings import *

def Monty_Model_softmax(X_set,model_test,batch_size):
    SoftMaxBin_list=[]
    for i in range(n_MoteCarlo_Samples):
        SoftMaxBin_list.append(model_test.predict_generator(X_set,batch_size=batch_size))
    SoftMaxBin=np.array(SoftMaxBin_list)
    
    SoftmaxMean=np.mean(SoftMaxBin,axis=0)
    SoftmaxStd=np.std(SoftMaxBin,axis=0)
#    print(SoftMaxBin.shape)
#    print(np.amin(SoftmaxStd))
#    print(np.amax(SoftmaxStd))
    return SoftmaxMean,SoftmaxStd

def predict_softmax(SoftmaxMean):
    class_predict=np.argmax(SoftmaxMean,axis=-1)
    return class_predict


## Model Evaluation function

def Models_eval_hands_B(model_filename_save,model_id,dataset):
    
    Keras_model=load_model(model_filename_save,custom_objects={'custom_loss_Bayesian': custom_loss_Bayesian})
    
    prc_bin=[]
    rec_bin=[]
    f1_bin=[]
    PRCurve_bin=[]
    mean_prc_bin=[]

    SoftmaxMean,SoftmaxStd=Monty_Model_softmax(X_set,Keras_model)   

    if model_id=='UW-MAP':
        # UW-MAP
        Y_prd_set_keras=predict_softmax(SoftmaxMean)
        Y_prd_set_UW_MAP=np.reshape(Y_prd_set_keras,(-1,X_set.shape[1],X_set.shape[2]))

        
        prd_set=Y_prd_set_UW_MAP
       
        prc_UW_MAP, rec_UW_MAP, f1s_UW_MAP, _ =precision_recall_fscore_support(Y_set.ravel(), Y_prd_set_UW_MAP.ravel())
        # prc_plt_UW_MAP, rec_plt_UW_MAP, _ = precision_recall_curve(Y_set.ravel(), Y_prob_set_UW_MAP.ravel())
        # mean_prc_UW_MAP = average_precision_score(Y_set.ravel(), Y_prob_set_UW_MAP.ravel())
        
        prc_bin=prc_UW_MAP
        rec_bin=rec_UW_MAP
        f1_bin=f1s_UW_MAP
        PRCurve_bin=[0,0]
        mean_prc_bin=0
        
    elif model_id=='UW-ML':
        # UW_ML
        Pk_thresh=1    
    
        Soft_max=SoftmaxMean
        Pk=pixel_count_node/X_train.shape[0]
        Pk[Pk==0]=Pk_thresh
        Pk=np.reshape(Pk,(1,Pk.shape[0],Pk.shape[1]))
        Prob_ML=Soft_max/Pk
        Pred_ML_vec=np.argmax(Prob_ML,axis=2)
        Y_prd_set_UW_ML=np.reshape(Pred_ML_vec,(-1,X_set.shape[1],X_set.shape[2]))
        
        prd_set=Y_prd_set_UW_ML
        
        prc_UW_ML, rec_UW_ML, f1s_UW_ML, _ =precision_recall_fscore_support(Y_set.ravel(), Y_prd_set_UW_ML.ravel())
        # prc_plt_UW_ML, rec_plt_UW_ML, _ = precision_recall_curve(Y_set.ravel(), Y_prob_set_UW_ML.ravel())
        # mean_prc_UW_ML = average_precision_score(Y_set.ravel(), Y_prob_set_UW_ML.ravel())
        
        prc_bin=prc_UW_ML
        rec_bin=rec_UW_ML
        f1_bin=f1s_UW_ML
        PRCurve_bin=[0,0]
        mean_prc_bin=0
    
    elif model_id=='MFW-MAP':
        # MFW-MAP
        Y_prd_set_keras=predict_softmax(SoftmaxMean)
        Y_prd_set_MFW_MAP=np.reshape(Y_prd_set_keras,(-1,X_set.shape[1],X_set.shape[2]))
        
        prd_set=Y_prd_set_MFW_MAP
        
        prc_MFW_MAP, rec_MFW_MAP, f1s_MFW_MAP, _ =precision_recall_fscore_support(Y_set.ravel(), Y_prd_set_MFW_MAP.ravel())
        # prc_plt_MFW_MAP, rec_plt_MFW_MAP, _ = precision_recall_curve(Y_set.ravel(), Y_prob_set_MFW_MAP.ravel())
        # mean_prc_MFW_MAP = average_precision_score(Y_set.ravel(), Y_prob_set_MFW_MAP.ravel())
        
        prc_bin=prc_MFW_MAP
        rec_bin=rec_MFW_MAP
        f1_bin=f1s_MFW_MAP
        PRCurve_bin=[0,0]
        mean_prc_bin=0
    
    else:
        print('Invalid model ID')

    SoftmaxStd=np.reshape(SoftmaxStd,(-1,X_set.shape[1],X_set.shape[2],n_classes))
    SoftmaxMean=np.reshape(SoftmaxMean,(-1,X_set.shape[1],X_set.shape[2],n_classes))
    
    p_c=SoftmaxMean
    max_p=np.sum(SoftmaxMean,axis=-1)
    max_p=np.reshape(max_p,max_p.shape+(1,))
    p_c_norm=np.divide(p_c,max_p)
    log_p_c=np.log10(p_c_norm+1e-12)
    entropy=np.sum(np.multiply(-1*p_c_norm,log_p_c),axis=-1)
    
    
    # Measure IoU
    
    classes=np.unique(Y_set)
    nClasses=len(classes)
    IoU_bin=[]

    Mask_GT=np.array(Y_set)
    Mask_Prd=np.array(prd_set) 

    for i in range(nClasses):
        
        GT_class_cond=(Mask_GT==classes[i])
        Prd_class_cond=(Mask_Prd==classes[i])
        # Measure IoU
        intersection=np.logical_and(GT_class_cond,Prd_class_cond)
        union=np.logical_or(GT_class_cond,Prd_class_cond)
        
        IoU_class = np.sum(intersection) / np.sum(union) 
        IoU_bin.append(IoU_class)    


    return prd_set,prc_bin,rec_bin,f1_bin,IoU_bin,PRCurve_bin,mean_prc_bin,SoftmaxStd,SoftmaxMean,entropy
