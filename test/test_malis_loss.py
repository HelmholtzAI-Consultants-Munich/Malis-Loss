# Unit Tests for this malis loss implementation - Tensorflow version and Pytoch version
# Given different inputs, check whether the outputs are right

import numpy as np
import malis
from malis.malis_keras import pairs_to_loss_keras
from malis.malis_torch import torchloss,pairs_to_loss_torch
import tensorflow as tf
import torch
from torch.autograd import grad

na = np.newaxis

#######################################################
# TEST 1
print("Starting test 1")
# the following variables are named the same as the ones in
# Turaga et al. 2009 (Malis paper)
x = 0.4
x_hat = 0.0
m = 0.3
pos_loss_weight = 0.5
neg_loss_weight = 0.5

labels = np.array([[[1, 1, 1, 2, 2, 2]]], dtype=np.int32)
affinities = np.zeros(shape=(3,) + labels.shape)
affinities[2, :, :, :] = np.array([[[.6, .6, .6, x, .6, .6]]], dtype=np.float32)
affinities += np.random.normal(size=affinities.shape, scale=.0001)

### keras/tensorflow (channel last)
labels_tf = tf.convert_to_tensor(labels)
affinities_tf = tf.convert_to_tensor(affinities,dtype=tf.float32)
pos_pairs, neg_pairs = tf.numpy_function(func = malis.get_pairs,inp=[labels_tf, affinities_tf],
                                             Tout=[tf.uint64,tf.uint64])
pos_pairs = tf.cast(pos_pairs,tf.float32)
neg_pairs = tf.cast(neg_pairs,tf.float32) 

with tf.GradientTape() as g:
    g.watch(pos_pairs)
    g.watch(neg_pairs)
    g.watch(affinities_tf)
    loss_tf = pairs_to_loss_keras(pos_pairs, neg_pairs, affinities_tf, pos_loss_weight=0.5)
    
gradient_tf = g.gradient(loss_tf, affinities_tf)

### pytorch (channel first)
affinities_torch = torch.tensor(affinities,dtype=torch.float32,requires_grad = True)
labels_torch = torch.from_numpy(labels)

pos_pairs,neg_pairs = torchloss.apply(affinities_torch, labels_torch)
loss_torch = pairs_to_loss_torch(pos_pairs, neg_pairs, affinities_torch, pos_loss_weight=0.5)
gradient_torch = grad(outputs=loss_torch, inputs=affinities_torch)[0]

# analytically expected loss:
pos_loss = pos_loss_weight * ((3 + 3) * (1 - 0.6 - m) ** 2)
neg_loss = neg_loss_weight * ((3 * 3) * (x - m) ** 2)
analytic_loss = (pos_loss + neg_loss) * 2
analytic_gradient = 2 * 9 * (0.4 - m)

print("The keras/tensorflow computed loss is: " + str(loss_tf.numpy()))
print("The pytorch computed loss is: " + str(loss_torch.detach().numpy()))
print("The analytically expected loss is: " + str(analytic_loss))

assert np.isclose(analytic_loss, loss_tf.numpy(), atol=0.01), "analytic loss and tensorflow loss not the same"
assert np.isclose(analytic_loss, loss_torch.detach().numpy(), atol=0.01), "analytic loss and pytorch loss not the same"

gradient_tf = gradient_tf[2, 0, 0, 3]
gradient_torch = gradient_torch[2, 0, 0, 3]
print("The keras/tensorflow computed gradient at the 'border' affinity is: " + str(gradient_tf.numpy()))
print("The pytorch computed gradient at the 'border' affinity is: " + str(gradient_torch.numpy()))
print("The analytically expected gradient is: " + str(analytic_gradient))

assert np.isclose(analytic_gradient, gradient_tf.numpy(), atol=0.1),"the gradient of tensorflow loss not correct"
assert np.isclose(analytic_gradient, gradient_torch.numpy(), atol=0.1),"the gradient of pytorch loss not correct"

print("Test 1 finished, no error")


########################################################
## TEST 2
print("\nTest 2")

# setting the 'border'-affinity to 0.1, which is below the margin, so there should be
# no loss for non-matching pairs and the gradient at this edge should be zero
affinities[2, 0, 0, 3] = 0.1

### keras/tensorflow 
labels_tf = tf.convert_to_tensor(labels)
affinities_tf = tf.convert_to_tensor(affinities,dtype=tf.float32)
pos_pairs, neg_pairs = tf.numpy_function(func = malis.get_pairs,inp=[labels_tf, affinities_tf],
                                             Tout=[tf.uint64,tf.uint64])
pos_pairs = tf.cast(pos_pairs,tf.float32)
neg_pairs = tf.cast(neg_pairs,tf.float32) 

with tf.GradientTape() as g:
    g.watch(pos_pairs)
    g.watch(neg_pairs)
    g.watch(affinities_tf)
    loss_tf = pairs_to_loss_keras(pos_pairs, neg_pairs, affinities_tf, pos_loss_weight=0.5)
    
gradient_tf = g.gradient(loss_tf, affinities_tf)

### pytorch
affinities_torch = torch.tensor(affinities,dtype=torch.float32,requires_grad = True)
labels_torch = torch.from_numpy(labels)

pos_pairs,neg_pairs = torchloss.apply(affinities_torch, labels_torch)
loss_torch = pairs_to_loss_torch(pos_pairs, neg_pairs, affinities_torch, pos_loss_weight=0.5)
gradient_torch = grad(outputs=loss_torch, inputs=affinities_torch)[0]


# analytically expected loss:
analytic_loss = (3 + 3) * (1 - 0.6 - m) ** 2 + 0
analytic_gradient = 0

print("The keras/tensorflow computed loss is: " + str(loss_tf.numpy()))
print("The pytorch computed loss is: " + str(loss_torch.detach().numpy()))
print("The analytically expected loss is: " + str(analytic_loss))
assert np.isclose(analytic_loss, loss_tf.numpy(), atol=0.01), "analytic loss and tensorflow loss not the same"
assert np.isclose(analytic_loss, loss_torch.detach().numpy(), atol=0.01), "analytic loss and pytorch loss not the same"

gradient_tf = gradient_tf[2, 0, 0, 3]
gradient_torch = gradient_torch[2, 0, 0, 3]
print("The keras/tensorflow computed gradient at the 'border' affinity is: " + str(gradient_tf.numpy()))
print("The pytorch computed gradient at the 'border' affinity is: " + str(gradient_torch.numpy()))
print("The analytically expected gradient is: " + str(analytic_gradient))
assert np.isclose(analytic_gradient, gradient_tf.numpy(), atol=0.1),"the gradient of tensorflow loss not correct"
assert np.isclose(analytic_gradient, gradient_torch.numpy(), atol=0.1),"the gradient of pytorch loss not correct"


print("Test 2 finished, no error")


