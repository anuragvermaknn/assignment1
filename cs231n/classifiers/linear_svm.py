import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #print "W.shape", W.shape
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    count_positive_max = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #print dW[:,j].shape, X[i].shape
        dW[:,j] += X[i]
        count_positive_max += 1
    dW[:,y[i]] -= (X[i] * count_positive_max)
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW

def max_func(x):
    return (x if x > 0 else 0)

def indicator_func(x):
    return (1 if x > 0 else 0)

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  num_train = X.shape[0]
  dW = np.zeros(W.shape) # initialize the gradient as zero
  Scores = X.dot(W)
  num_classes = W.shape[1]
  Syi_idxs = [np.arange(X.shape[0]), y ]
  
  Scores_yi = Scores[Syi_idxs] 
  print "Scores_yi.shape", Scores_yi.shape
    
  svm_loss =   Scores - Scores_yi.reshape(1,Scores_yi.shape[0]).T + 1
  np_max_func = np.vectorize(max_func)
  loss = np_max_func(svm_loss)
  print "loss.shape", loss.T.shape
  #loss = loss.T
  
  print "loss",loss
  loss[Syi_idxs] = 0
  print "loss",loss
  
  loss = np.sum(loss, axis = 1)
  loss = np.sum(loss, axis = 0)
  print "loss", loss
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  np_indicator_func = np.vectorize(indicator_func)
  binary_svm_loss = np_indicator_func(svm_loss)
  binary_svm_loss[Syi_idxs] = 0
  print "binary_svm_loss.shape", binary_svm_loss.shape
  binary_svm_loss_yi = np.sum(binary_svm_loss, axis = 1) * (-1)
  binary_svm_loss[Syi_idxs] = binary_svm_loss_yi
  print "X.reshape(X.shape[1], X.shape[0], 1).shape", X.reshape(X.shape[1], X.shape[0], 1).shape
  dervatives = np.multiply( X.reshape(X.shape[1], X.shape[0], 1), binary_svm_loss)
  print "dervatives.shape",dervatives.shape
  print "dW.shape", dW.shape
  dW = np.sum(dervatives, axis =1)
  print "dW.shape", dW.shape
  
  dW += reg*W
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
