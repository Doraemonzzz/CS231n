import numpy as np
from random import shuffle
from past.builtins import xrange

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
  N, D = X.shape
  for i in range(N):
      x = X[i]
      score = x.dot(W)
      c = y[i]
      #计算概率
      p = np.exp(score) / np.sum(np.exp(score))
      #计算损失
      loss += - np.log(p[c])
      #计算梯度第一项
      dW[:, y[i]] -= x
      #计算梯度第二项
      dW += x.reshape(-1, 1).dot(p.reshape(1, -1))
  
  #计算最终损失
  loss /= N
  loss += reg * np.sum(W * W)
  #计算最终梯度
  dW /= N
  dW += 2 * reg * W
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N, D = X.shape
  S = X.dot(W)
  #计算概率
  p1 = np.exp(S[np.arange(N), y])
  p2 = np.sum(np.exp(S), axis=1)
  p = p1 / p2
  #损失
  loss = np.sum(-np.log(p)) / N + reg * np.sum(W * W)
  #梯度
  p3 = np.exp(S) / p2.reshape(-1, 1)
  p3[np.arange(N), y] -= 1
  dW = X.T.dot(p3) / N + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

