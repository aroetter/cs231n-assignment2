import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net.
    ############################################################################
    ## Implement with a FC network
    self.model = FullyConnectedNet([hidden_dim], input_dim, num_classes,
                                   reg=reg, weight_scale=weight_scale,
                                   dtype=np.float64)

    ## This is silly, it's *only* needed so that the "checking" code that
    ## makes sure I implemented this right, from FullyConnectedNets.ipynb,
    ## works, b/c it expects this class to have a params member
    self.params = self.model.params

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net
    ############################################################################
    return self.model.loss(X, y)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    all_dims = [input_dim] + hidden_dims + [num_classes]
    for idx in xrange(self.num_layers):
      nrows = all_dims[idx]
      ncols = all_dims[idx+1]
      
      layer_name = "%d" % (idx+1)
      weight_name = "W" + layer_name
      bias_name = "b" + layer_name
      self.params[weight_name] = weight_scale * np.random.randn(nrows, ncols)
      self.params[bias_name] = np.zeros(ncols)

      if use_batchnorm and idx < (self.num_layers - 1):
        self.params["gamma" + layer_name] = np.ones(ncols)
        self.params["beta" + layer_name] = np.zeros(ncols)
      pass

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    caches = {}
    scores = X
    
    # iterate over every layer...
    for i in xrange(1, self.num_layers + 1):
      layer_name = "%d" % i
      W_name = "W" + layer_name
      b_name = "b" + layer_name
      gamma_name = "gamma" + layer_name
      beta_name = "beta" + layer_name
      dropout_name = "dropout" + layer_name
      cache_name = "cache" + layer_name
      
      if self.num_layers == i:
        # HANDLE FINAL LAYER HERE THAT IS AFFINE ONLY
        scores, cache = affine_forward(scores, self.params[W_name],
                                       self.params[b_name])
        pass
      else:
        # Regular hidden layer here
        if self.use_batchnorm:
          scores, cache = affine_batchnorm_relu_forward(
            scores, self.params[W_name], self.params[b_name],
            self.params[gamma_name], self.params[beta_name], self.bn_params[i-1])
        else:
          scores, cache = affine_relu_forward(
            scores, self.params[W_name], self.params[b_name])
        # Optionally do dropout here
        if self.use_dropout:
          scores, caches[dropout_name] = dropout_forward(scores, self.dropout_param)
  
      caches[cache_name] = cache
      pass
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    grads = {}
    loss, der = softmax_loss(scores, y)
    
    #### affine_backward(dout, cache) ==> dx, dw, db
    #### relu_backward(dout, cache) ==> dx
    #### batchnorm_backward(dout, cache) ==> dx, dgamma, dbeta

    for i in xrange(self.num_layers, 0, -1):
      # get all the param names
      layer_name = "%d" % i
      W_name = "W" + layer_name
      b_name = "b" + layer_name
      gamma_name = "gamma" + layer_name
      beta_name = "beta" + layer_name
      dropout_name = "dropout" + layer_name
      cache_name = "cache" + layer_name
      
      # L2 regularization loss
      loss += 0.5 * self.reg * (self.params[W_name] ** 2).sum()
      
      if(self.num_layers == i):
        # final layer
        der, grads[W_name], grads[b_name] = affine_backward(
          der, caches[cache_name])
      else:
        # Optional dropout
        if self.use_dropout:
          der = dropout_backward(der, caches[dropout_name])
        
        if self.use_batchnorm:
          (der, grads[W_name], grads[b_name], grads[gamma_name],
           grads[beta_name]) = affine_batchnorm_relu_backward(
             der, caches[cache_name])
        else:
          (der, grads[W_name], grads[b_name]
          ) = affine_relu_backward(der, caches[cache_name])

      # the loss depends on the reg term, so update gradient accordingly
      grads[W_name] += self.reg * self.params[W_name]


          
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
