import numpy as np
from random import shuffle
import math

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  for i in range(num_train):
    
    scores = X[i].dot(W)  # dimC
    
    # log probabilities
    # first shift the values of f so that the highest number is 0 to avoid numeric problems
    scores -= np.max(scores) 
    
    # probabilities are P(y=k | x=xi) = e^s_k / sum (e^s_k) 
    softmax_prob = np.exp(scores) / np.sum(np.exp(scores))
    
    # Loss is log of the correct class
    loss += -np.log(softmax_prob[y[i]])
    
    for j in range(num_classes):
      dW[:,j] += X[i] * softmax_prob[j]
    dW[:,y[i]] += -X[i]
    
  loss /= num_train
  dW /= num_train
    
  loss += reg * np.sum(W*W)
  dW += reg * 2*W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def softmax_loss_m2(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  for i in range(num_train):
    score =[0]*num_classes
    for j in range(num_classes):
        dot = np.dot(X[i],W[:,j])
        norm = np.linalg.norm(X[i]) * np.linalg.norm(W[j])
        angle = dot / norm
        score[j] = norm * angle
        if j == y[i]:
            angle = 2* (angle)**2 -1 
            score[j] = norm * angle
    scores = score
    #scores = X[i].dot(W)  # dimC
    
    # log probabilities
    # first shift the values of f so that the highest number is 0 to avoid numeric problems
    scores -= np.max(scores) 
    
    # probabilities are P(y=k | x=xi) = e^s_k / sum (e^s_k) 
    softmax_prob = np.exp(scores) / np.sum(np.exp(scores))
    
    # Loss is log of the correct class
    loss += -np.log(softmax_prob[y[i]])
    
    for j in range(num_classes):
      dW[:,j] += X[i] * softmax_prob[j]
    dW[:,y[i]] += -X[i]
    
  loss /= num_train
  dW /= num_train
    
  loss += reg * np.sum(W*W)
  dW += reg * 2*W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def softmax_loss_m3(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  for i in range(num_train):
    score =[0]*num_classes
    for j in range(num_classes):
        dot = np.dot(X[i],W[:,j])
        norm = np.linalg.norm(X[i]) * np.linalg.norm(W[j])
        angle = dot / norm
        score[j] = norm * angle
        if j == y[i]:
            angle = (4*(angle)**3) - (3*(angle))
            score[j] = norm * angle
    scores = score
    #scores = X[i].dot(W)  # dimC
    
    # log probabilities
    # first shift the values of f so that the highest number is 0 to avoid numeric problems
    scores -= np.max(scores) 
    
    # probabilities are P(y=k | x=xi) = e^s_k / sum (e^s_k) 
    softmax_prob = np.exp(scores) / np.sum(np.exp(scores))
    
    # Loss is log of the correct class
    loss += -np.log(softmax_prob[y[i]])
    
    for j in range(num_classes):
      dW[:,j] += X[i] * softmax_prob[j]
    dW[:,y[i]] += -X[i]
    
  loss /= num_train
  dW /= num_train
    
  loss += reg * np.sum(W*W)
  dW += reg * 2*W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_m4(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  for i in range(num_train):
    score =[0]*num_classes
    for j in range(num_classes):
        dot = np.dot(X[i],W[:,j])
        norm = np.linalg.norm(X[i]) * np.linalg.norm(W[j])
        angle = dot / norm
        score[j] = norm * angle
        if j == y[i]:
            angle = (8*(angle)**4) - (8*(angle)**2) +1 
            score[j] = norm * angle
    scores = score
    #scores = X[i].dot(W)  # dimC
    
    # log probabilities
    # first shift the values of f so that the highest number is 0 to avoid numeric problems
    scores -= np.max(scores) 
    
    # probabilities are P(y=k | x=xi) = e^s_k / sum (e^s_k) 
    softmax_prob = np.exp(scores) / np.sum(np.exp(scores))
    
    # Loss is log of the correct class
    loss += -np.log(softmax_prob[y[i]])
    
    for j in range(num_classes):
      dW[:,j] += X[i] * softmax_prob[j]
    dW[:,y[i]] += -X[i]
    
  loss /= num_train
  dW /= num_train
    
  loss += reg * np.sum(W*W)
  dW += reg * 2*W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def asoftmax(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  for i in range(num_train):
    score =[0]*num_classes
    for j in range(num_classes):
        dot = np.dot(X[i],W[:,j])
        #norm = np.linalg.norm(X[i]) * np.linalg.norm(W[j])
        norm = np.linalg.norm(X[i])
        angle = dot / norm
        score[j] = norm * angle
        
    scores = score
    #scores = X[i].dot(W)  # dimC
    
    # log probabilities
    # first shift the values of f so that the highest number is 0 to avoid numeric problems
    scores -= np.max(scores) 
    
    # probabilities are P(y=k | x=xi) = e^s_k / sum (e^s_k) 
    softmax_prob = np.exp(scores) / np.sum(np.exp(scores))
    
    # Loss is log of the correct class
    loss += -np.log(softmax_prob[y[i]])
    
    for j in range(num_classes):
      dW[:,j] += X[i] * softmax_prob[j]
    dW[:,y[i]] += -X[i]
    
  loss /= num_train
  dW /= num_train
    
  loss += reg * np.sum(W*W)
  dW += reg * 2*W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def asoftmax_m2(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  for i in range(num_train):
    score =[0]*num_classes
    for j in range(num_classes):
        dot = np.dot(X[i],W[:,j])
        #norm = np.linalg.norm(X[i]) * np.linalg.norm(W[j])
        norm = np.linalg.norm(X[i])
        angle = dot / norm
        score[j] = norm * angle
        if j == y[i]:
            angle = 2* (angle)**2 -1 
            score[j] = norm * angle
    scores = score
    #scores = X[i].dot(W)  # dimC
    
    # log probabilities
    # first shift the values of f so that the highest number is 0 to avoid numeric problems
    scores -= np.max(scores) 
    
    # probabilities are P(y=k | x=xi) = e^s_k / sum (e^s_k) 
    softmax_prob = np.exp(scores) / np.sum(np.exp(scores))
    
    # Loss is log of the correct class
    loss += -np.log(softmax_prob[y[i]])
    
    for j in range(num_classes):
      dW[:,j] += X[i] * softmax_prob[j]
    dW[:,y[i]] += -X[i]
    
  loss /= num_train
  dW /= num_train
    
  loss += reg * np.sum(W*W)
  dW += reg * 2*W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def asoftmax_m3(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  for i in range(num_train):
    score =[0]*num_classes
    for j in range(num_classes):
        dot = np.dot(X[i],W[:,j])
        #norm = np.linalg.norm(X[i]) * np.linalg.norm(W[j])
        norm = np.linalg.norm(X[i])
        angle = dot / norm
        score[j] = norm * angle
        if j == y[i]:
            angle = (4*(angle)**3) - (3*(angle))
            score[j] = norm * angle
    scores = score
    #scores = X[i].dot(W)  # dimC
    
    # log probabilities
    # first shift the values of f so that the highest number is 0 to avoid numeric problems
    scores -= np.max(scores) 
    
    # probabilities are P(y=k | x=xi) = e^s_k / sum (e^s_k) 
    softmax_prob = np.exp(scores) / np.sum(np.exp(scores))
    
    # Loss is log of the correct class
    loss += -np.log(softmax_prob[y[i]])
    
    for j in range(num_classes):
      dW[:,j] += X[i] * softmax_prob[j]
    dW[:,y[i]] += -X[i]
    
  loss /= num_train
  dW /= num_train
    
  loss += reg * np.sum(W*W)
  dW += reg * 2*W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def asoftmax_m4(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  for i in range(num_train):
    score =[0]*num_classes
    for j in range(num_classes):
        dot = np.dot(X[i],W[:,j])
        #norm = np.linalg.norm(X[i]) * np.linalg.norm(W[j])
        norm = np.linalg.norm(X[i])
        angle = dot / norm
        score[j] = norm * angle
        if j == y[i]:
            angle = (8*(angle)**4) - (8*(angle)**2) +1 
            score[j] = norm * angle
    scores = score
    #scores = X[i].dot(W)  # dimC
    
    # log probabilities
    # first shift the values of f so that the highest number is 0 to avoid numeric problems
    scores -= np.max(scores) 
    
    # probabilities are P(y=k | x=xi) = e^s_k / sum (e^s_k) 
    softmax_prob = np.exp(scores) / np.sum(np.exp(scores))
    
    # Loss is log of the correct class
    loss += -np.log(softmax_prob[y[i]])
    
    for j in range(num_classes):
      dW[:,j] += X[i] * softmax_prob[j]
    dW[:,y[i]] += -X[i]
    
  loss /= num_train
  dW /= num_train
    
  loss += reg * np.sum(W*W)
  dW += reg * 2*W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW



def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.max(scores, axis=1, keepdims=True) 
  
  softmax_prob = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

  count = np.array(list(range(num_train)))  
  
  loss = np.sum(-np.log(softmax_prob[count,y]))
  # average loss
  loss /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W) 
  
  softmax_prob[count,y] -= 1 
  # x is (N*D) - transpose (D*N), svm_prom (N*C)
  dW = np.matmul(X.T, softmax_prob)    #(D*C)
    
  dW /= num_train
  dW += reg * 2 * W  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

