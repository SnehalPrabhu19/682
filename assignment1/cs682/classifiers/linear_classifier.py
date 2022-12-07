from __future__ import print_function

import numpy as np
from cs682.classifiers.linear_svm import *
from cs682.classifiers.softmax import *

class LinearClassifier(object):

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      # lazily initialize W
      self.W = 0.001 * np.random.randn(dim, num_classes)

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      batch_index = np.random.choice(num_train, batch_size)
      X_batch = X[batch_index]
      y_batch = y[batch_index]
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
      self.W -= grad * learning_rate
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[0])
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    score = np.matmul(X,self.W) # N*C
    
    y_pred = np.argmax(score, axis=1)
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred

  def loss(self, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative.
    Subclasses will override this.

    Inputs:
    - X_batch: A numpy array of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
    pass


class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)

class SVM_softmax(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X_batch, y_batch, reg):
    svm_loss, dw_svm = svm_loss_vectorized(self.W, X_batch, y_batch, reg)  
    sm_loss, dw_sm = softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
    loss_total = svm_loss + sm_loss
    dw_total = dw_svm + dw_sm
    return loss_total,dw_total 


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_naive(self.W, X_batch, y_batch, reg)

class Softmax_m2(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg): 
    return softmax_loss_m2(self.W, X_batch, y_batch, reg)

class Softmax_m3(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg): 
    return softmax_loss_m3(self.W, X_batch, y_batch, reg)


class Softmax_m4(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg): 
    return softmax_loss_m4(self.W, X_batch, y_batch, reg)

class ASoftmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg): 
    return asoftmax(self.W, X_batch, y_batch, reg)

class ASoftmax_m2(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg): 
    return asoftmax_m2(self.W, X_batch, y_batch, reg)

class ASoftmax_m3(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg): 
    return asoftmax_m3(self.W, X_batch, y_batch, reg)

class ASoftmax_m4(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg): 
    return asoftmax_m4(self.W, X_batch, y_batch, reg)


class ASoftmax_Lsoftmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    loss_m3, dw_m3 = softmax_loss_m3(self.W, X_batch, y_batch, reg)
    loss_am3, dw_am3 = asoftmax_m3(self.W, X_batch, y_batch, reg)
    loss_total = loss_m3 + loss_am3
    dw_total = dw_m3 + dw_am3

    return  loss_total,dw_total 
    

class Softmax_combine(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    loss_m2, dw_m2 = softmax_loss_m2(self.W, X_batch, y_batch, reg)
    loss_vector, dw_vector = softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
    loss_total = loss_m2 + loss_vector
    dw_total = dw_m2 + dw_vector

    return  loss_total,dw_total 
    
class Softmax_combinem3(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    loss_m2, dw_m2 = softmax_loss_m3(self.W, X_batch, y_batch, reg)
    loss_vector, dw_vector = softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
    loss_total = loss_m2 + loss_vector
    dw_total = dw_m2 + dw_vector

    return  loss_total,dw_total 

class Softmax_combinem4(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    loss_m2, dw_m2 = softmax_loss_m4(self.W, X_batch, y_batch, reg)
    loss_vector, dw_vector = softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
    loss_total = loss_m2 + loss_vector
    dw_total = dw_m2 + dw_vector

    return  loss_total,dw_total 
    
class Softmax_m2m3(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    loss_m2, dw_m2 = softmax_loss_m2(self.W, X_batch, y_batch, reg)
    loss_m3, dw_m3 = softmax_loss_m3(self.W, X_batch, y_batch, reg)
    loss_total = loss_m2 + loss_m3
    dw_total = dw_m2 + dw_m3

    return  loss_total,dw_total 

class Softmax_m3m4(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    loss_m4, dw_m4 = softmax_loss_m4(self.W, X_batch, y_batch, reg)
    loss_m3, dw_m3 = softmax_loss_m3(self.W, X_batch, y_batch, reg)
    loss_total = loss_m4 + loss_m3
    dw_total = dw_m4 + dw_m3

    return  loss_total,dw_total 

class Softmax_m2m4(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    loss_m2, dw_m2 = softmax_loss_m2(self.W, X_batch, y_batch, reg)
    loss_m4, dw_m4 = softmax_loss_m4(self.W, X_batch, y_batch, reg)
    loss_total = loss_m2 + loss_m4
    dw_total = dw_m2 + dw_m4

    return  loss_total,dw_total 
