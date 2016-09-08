import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    # each size is prepended by a N dimension for N data points. I omit that
    # in working out / documenting the sizes below.
    #input: is of size C, H, W
    # conv (params are W1, b1): shape is now N, F, H, W (we rely on fact that HxW
    # are unchanged by the conv layer.
    # relu: shape is unchanged= F, H, W
    # max_pool (2x2 max). shape is now F, H/2, W/2
    # affine (params are W2(hidden_dim, F, H/2, W/2) , b2(hidden_dim)).
    #   shape is now:  (hidden_dim)
    # relu. shape is now hidden_dim.
    # affine (W3(num_classes,hidden_dim), b3(num_classes).
    #   shape is num_classes * hidden_dim)
    C, H, W = input_dim
    self.params['W1'] = weight_scale * np.random.randn(
      num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros([num_filters])

    # really W2 should be hidden_dim x num_filters x H/2 x W/2 but i need
    # to flatten last three to make the dot products work (which assume 2D
    # weight matrixes, and also transpose the order of dims here.
    self.params['W2'] = weight_scale * np.random.randn(
      num_filters * H/2 * W/2, hidden_dim)
    self.params['b2'] = np.zeros([hidden_dim])

    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros([num_classes])
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    # this conv_param guarantees that W' = W, that is, conv layer doesn't
    # change H x W, we rely on this fact above in the initialization
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    caches = {}

    (scores, caches['conv']) = conv_relu_pool_forward(
      X, W1, b1, conv_param, pool_param)
    (scores, caches['hidden']) = affine_relu_forward(scores, W2, b2)
    (scores, caches['final']) = affine_forward(scores, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    grads = {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, der = softmax_loss(scores, y)

    der, grads['W3'], grads['b3'] = affine_backward(der, caches['final'])
    der, grads['W2'], grads['b2'] = affine_relu_backward(der, caches['hidden'])
    der, grads['W1'], grads['b1'] = conv_relu_pool_backward(der, caches['conv'])

    # add in L2 loss for W1, W2, W3
    for w_name in ['W1', 'W2', 'W3']:
      loss += 0.5 * self.reg * (self.params[w_name] ** 2).sum()
      # the gradient of the loss wrt the Ws also changes b/c of regularization
      grads[w_name] += self.reg * self.params[w_name]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return loss, grads
  
  
pass
