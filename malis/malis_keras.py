import keras.backend as K
import tensorflow as tf
import numpy as np
from .wrappers import get_pairs
from .pairs_cython import mknhood3d


def pairs_to_loss_keras(pos_pairs, neg_pairs, pred, margin=0.3, pos_loss_weight=0.3):
    """
    Computes MALIS loss weights from given positive and negtive weights.
    
    Roughly speaking the malis weights quantify the impact of an edge in
    the predicted affinity graph on the resulting segmentation.
    
    Input:
        pos_pairs: (batch_size, H, W, C)
           Contains the positive pairs 
        neg_pairs: (batch_size, H, W, C)
           Contains the negative pairs 
        pred:  (batch_size, H, W, C)
            affinity predictions from network
    Returns:
        malis_loss: scale
            final malis loss
        
    """
    neg_loss_weight = 1 - pos_loss_weight
    zeros_helpervar = tf.zeros(shape=tf.shape(pred))

    pos_loss = tf.where(1 - pred - margin > 0,
                        (1 - pred - margin)**2,
                        zeros_helpervar)
    pos_loss = pos_loss * pos_pairs
    pos_loss = tf.reduce_sum(pos_loss) * pos_loss_weight

    neg_loss = tf.where(pred - margin > 0,
                        (pred - margin)**2,
                        zeros_helpervar)
    neg_loss = neg_loss * neg_pairs
    neg_loss = tf.reduce_sum(neg_loss) * neg_loss_weight
    malis_loss = (pos_loss + neg_loss) * 2  # because of the pos_loss_weight and neg_loss_weight

    return malis_loss

def malis_loss2d(y_true,y_pred): 
    '''
    Computes 2d MALIS loss given predicted affinity graphs and segmentation groundtruth
    
    Roughly speaking malis weights (pos_pairs and neg_pairs) quantify the 
    impact of an edge in the predicted affinity graph on the resulting segmentation.
    
    Input:
       y_true: Tensor (batch_size, H, W, C = 1)
          segmentation groundtruth
       y_pred: Tensor (batch_size, H, W, C = 2)
           affinity predictions from network
    Returns:
       loss: Tensor(scale)
              malis loss 
      
    Outline:
    - Computes for all pixel-pairs the MaxiMin-Affinity
    - Separately for pixel-pairs that should/should not be connected
    - Every time an affinity prediction is a MaxiMin-Affinity its weight is
      incremented by one in the output matrix (in different slices depending
      on whether that that pair should/should not be connected)
    '''
    
    #########  make sure seg_true and y_pred has the correct shape -> (H,W,C'), (2,H,W,batch) for each     
    x = K.int_shape(y_pred)[1]  # H
    y = K.int_shape(y_pred)[2]  # W

    seg_true = K.reshape(y_true,(x,y,-1))             # (H,W,C'=C*batch_size)
    y_pred = K.permute_dimensions(y_pred,(3,1,2,0))   # (C=2,H,W,batch_size)
    #########
    
    nhood = mknhood3d(1)[:-1]  # define neighborhood structure, check mknhood3d in pairs_cython.pyx for further information
                               # calculating connectivity among x and y axis here
    pos_pairs, neg_pairs = tf.numpy_function(func = get_pairs,inp=[seg_true, y_pred, nhood],
                                             Tout=[tf.uint64,tf.uint64])  # get positive and negtive malis weights
    pos_pairs = tf.cast(pos_pairs,tf.float32)
    neg_pairs = tf.cast(neg_pairs,tf.float32) 

    loss = pairs_to_loss_keras(pos_pairs, neg_pairs, y_pred)  # computes final malis loss
    
    return loss


def malis_loss3d(y_true,y_pred): 
    '''
    Computes 3d MALIS loss given predicted affinity graphs and segmentation groundtruth
    
    Roughly speaking malis weights (pos_pairs and neg_pairs) quantify the 
    impact of an edge in the predicted affinity graph on the resulting segmentation.
    
    Input:
       y_true: Tensor (batch_size=1, H, W, D, C=1)
          segmentation groundtruth
       y_pred: Tensor (batch_size=1, H, W, D, C=3)
           affinity predictions from network
    Returns:
       loss: Tensor(scale)
              malis loss 
    
    Outline:
    - Computes for all pixel-pairs the MaxiMin-Affinity
    - Separately for pixel-pairs that should/should not be connected
    - Every time an affinity prediction is a MaxiMin-Affinity its weight is
      incremented by one in the output matrix (in different slices depending
      on whether that that pair should/should not be connected)
    '''
    
    ######### make sure seg_true and y_pred has the correct shape -> (H,W,D),(3,H,W,D) for each   
    x = K.int_shape(y_pred)[1]  # H
    y = K.int_shape(y_pred)[2]  # W
    z = K.int_shape(y_pred)[3]  # D

    seg_true = K.reshape(y_true,(x,y,z))              # (H,W,D)
    y_pred = K.reshape(y_pred,(x,y,z,-1))             # (H,W,D,C=3)
    y_pred = K.permute_dimensions(y_pred,(3,0,1,2))   # (C=3,H,W,D)
    
    #########
    
    nhood = mknhood3d(1)   # define neighborhood structure, check mknhood3d in pairs_cython.pyx for further information
                           # calculating connectivity among x, y and z axis here               
    pos_pairs, neg_pairs = tf.numpy_function(func = get_pairs,inp=[seg_true, y_pred, nhood],
                                             Tout=[tf.uint64,tf.uint64]) # get positive and negtive malis weights
    pos_pairs = tf.cast(pos_pairs,tf.float32)
    neg_pairs = tf.cast(neg_pairs,tf.float32) 

    loss = pairs_to_loss_keras(pos_pairs, neg_pairs, y_pred) # computes final malis loss
    
    return loss