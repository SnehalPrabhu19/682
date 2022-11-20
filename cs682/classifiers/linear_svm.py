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

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)         #1*C
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # gradient is the partial derivative of loss w.r.t w = dl(Wx)/dw = x
        # dw dimensions (D*C)
        dW[:,j] += X[i]
        dW[:,y[i]] += -X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW += reg * 2 * W 

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)  #(N*C)
  
  # get the scores of the correct class for N images 
  #correct_scores = np.array([scores[i][y[i]] for i in range(num_train)])
  count = np.array(list(range(num_train)))  
  correct_scores = np.array(scores[count,y])      #(N*1)
  correct_scores = np.transpose([correct_scores])  #(1*N)
  
  # get the margin for all classes (N*C)  
  margin = np.maximum(0, scores - correct_scores + 1) # delta=1
  
  # ignore the loss of the correct class  
  margin[count,y] = 0
  
  # sum over all losses
  loss = np.sum(margin)
  # average loss
  loss /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # get all the margin>0 from above computation , (N*C)
  margin_val = (margin>0).astype(np.int_)  
  # the correct classes
  margin_val[count,y] -= np.sum(margin_val,axis=1)
  # x is (N*D) - transpose (D*N) 
  dW = np.matmul(X.T, margin_val)  # (D*C)
    
  dW /= num_train
  dW += reg * 2 * W 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
