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
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W


  #############################################################################
  # DONE:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # DONE:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_pixels = X.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  scores = X.dot(W)
  correct_class_scores = scores[range(num_train), y]
  incorrect_scores = np.ma.masked_equal(scores, correct_class_scores.reshape(num_train, -1))
  margins = incorrect_scores - correct_class_scores.reshape(num_train, -1) + 1
  positive_margins = np.ma.masked_less_equal(margins, 0)
  loss = np.sum(positive_margins) / num_train + 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # DONE:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  masked_dW = np.ma.masked_array(
    np.broadcast_to(X.reshape(num_train, num_pixels, 1), (num_train, num_pixels, num_classes)),
    np.broadcast_to(positive_margins.mask.reshape(num_train, 1, num_classes), (num_train, num_pixels, num_classes))
  )
  correct_X = np.ma.masked_array(
    np.broadcast_to(X.reshape(num_train, num_pixels, 1), (num_train, num_pixels, num_classes)),
    np.logical_not(np.broadcast_to(incorrect_scores.mask.reshape(num_train, 1, num_classes), (num_train, num_pixels, num_classes)))
  )

  negative_dW = correct_X * np.sum(np.logical_not(positive_margins.mask), 1).reshape(num_train, 1, 1)

  dW = np.sum(masked_dW, 0) - np.sum(negative_dW, 0) / num_train + reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
