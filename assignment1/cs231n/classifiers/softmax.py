import numpy as np
from random import shuffle

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
  for i in range(X.shape[0]):
      scores = X[i].dot(W)
      log_sum = 0
      for j in range(W.shape[1]):
          log_sum += np.exp(scores[j])
      loss += -scores[y[i]] + np.log(log_sum)
      for j in range(W.shape[1]):
        if j != y[i]:
          dW[:,j] += X[i] * np.exp(scores[j])/ log_sum
        else:
          dW[:,y[i]] -= X[i]*(1 - np.exp(scores[j])/ log_sum)
  loss /= X.shape[0]
  loss += reg * np.sum(W*W)
  dW /= X.shape[0]
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def to_binar(X, size = 10):
  from scipy.sparse import csr_matrix
  return csr_matrix(([1]*X.shape[0],
                    (np.arange(X.shape[0]), X)),
                     shape=(X.shape[0], size)).toarray()
def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores = np.exp(scores)
  scores = scores/scores.sum(axis = 1)[:,np.newaxis]
  loss = -(np.log(scores) * to_binar(y)).sum()
  loss /= X.shape[0]
  loss += reg * np.sum(W*W)
  dW = ((scores - to_binar(y)).T.dot(X).T)/X.shape[0]
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

