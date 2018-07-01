from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    next_h = x.dot(Wx) + prev_h.dot(Wh) + b
    next_h = np.tanh(next_h)
    cache = (x, next_h, prev_h, Wx, Wh, b)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    x, next_h, prev_h, Wx, Wh, b = cache
    dd =  dnext_h * (1 - next_h**2)
    dx, dprev_h, dWx, dWh, db = dd.dot(Wx.T), dd.dot(Wh.T),\
                                dd.T.dot(x).T, dd.T.dot(prev_h).T, dd.sum(axis = 0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    
    N, T, D = x.shape
    H = Wx.shape[1]
    h = np.zeros((T, N, H))
    x = x.swapaxes(1, 0)
    cache = []
    h[0], tmp = rnn_step_forward(x[0], h0, Wx, Wh, b)
    cache.append(tmp)
    for i in range(1, T):
        h[i], tmp = rnn_step_forward(x[i], h[i-1], Wx, Wh, b)
        cache.append(tmp)
    h = h.swapaxes(1, 0)
    cache.append(D)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    
    N, T, H = dh.shape
    D = cache[-1]
    dh = dh.swapaxes(1, 0)

    dx = np.zeros((T, N, D))
    dh0 = np.zeros((T, N, H))
    dWx = np.zeros((T, D, H))
    dWh = np.zeros((T, H, H))
    db = np.zeros((T, H))

    for i in reversed(range(T)):
        dx[i], dh0[i-1], dWx[i], dWh[i], db[i] =\
                                         rnn_step_backward(dh[i]+dh0[i], cache[i]) 
    dWx = dWx.sum(axis = 0)
    dWh = dWh.sum(axis = 0)
    dh0 = dh0[-1]
    db = db.sum(axis = 0) 
    dx = dx.swapaxes(1, 0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    out, cache= W[x[:,]], (x, W.shape)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x, S = cache
    dW = np.zeros(S)
    np.add.at(dW, x, dout) 

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    def sum_layer(wx, wh, b):
        return x @ wx + prev_h @ wh  + b
    H = Wx.shape[1]//4
    
    # packed weights
    def splitting(w):
        out = {}
        out['i'] = w[:,:H]
        out['f'] = w[:,H:2*H]
        out['o'] = w[:,2*H:3*H]
        out['g'] = w[:,3*H:]
        return out
    Wx = splitting(Wx)
    Wh = splitting(Wh)
    # input gate
    i_gate = sigmoid(sum_layer(Wx['i'], Wh['i'], b[:H]))
    # forget gate
    f_gate = sigmoid(sum_layer(Wx['f'], Wh['f'], b[H:2*H]))
    # output gate
    out_gate = sigmoid(sum_layer(Wx['o'], Wh['o'], b[2*H:3*H]))
    # gate-gate
    g_gate = np.tanh(sum_layer(Wx['g'], Wh['g'], b[3*H:]))
    # next cell state
    next_c = f_gate * prev_c + i_gate * g_gate
    # next hidden layer
    next_h = out_gate * np.tanh(next_c)
    # packed cache
    gates = (i_gate, f_gate, out_gate, g_gate)
    step = (prev_h, next_c, prev_c)
    cache = (x, step, gates, Wx, Wh, H)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    # unpack cache
    (x, step, gates, Wx, Wh, H) = cache
    i_gate, f_gate, out_gate, g_gate = gates
    prev_h, c, prev_c = step
    N, D = x.shape
    
    # zero initialize derivative
    dWx = np.zeros((D,4*H))
    dWh = np.zeros((H,4*H))
    db = np.zeros((4*H,))

    # (6): i_gate = sigmoid(sum_layer(Wx[:,:H], Wh[:,:H], b[:H]))
    # (5): f_gate = sigmoid(sum_layer(Wx[:,H:2*H], Wh[:,H:2*H], b[H:2*H]))
    # (4): out_gate = sigmoid(sum_layer(Wx[:,2*H:3*H], Wh[:,2*H:3*H], b[2*H:3*H]))
    # (3): g_gate = np.tanh(sum_layer(Wx[:,3*H:], Wh[:,3*H:], b[3*H:]))
    # (2): next_c = f_gate * prev_c + i_gate * g_gate
    # (1): next_h = out_gate * np.tanh(next_c)
    
    # derivatives :
    #(1):
    dWx[:,2*H:3*H] = ((out_gate * (1 - out_gate) * dnext_h * np.tanh(c)).T @ x).T
    dWh[:,2*H:3*H] = ((out_gate * (1 - out_gate) * dnext_h * np.tanh(c)).T @ prev_h).T
    db[2*H:3*H] = (out_gate * (1 - out_gate) * dnext_h * np.tanh(c)).sum(axis = 0)
    
    dx = (out_gate * (1 - out_gate) * dnext_h * np.tanh(c)) @ Wx['o'].T
    dprev_h = (out_gate * (1 - out_gate) * dnext_h * np.tanh(c)) @ Wh['o'].T
    # grad dnext_h plus dnext_c using chain rule in (2):
    dc = out_gate * dnext_h * (1 - np.tanh(c)**2) + dnext_c
    dprev_c =  f_gate * dc
    #(2),(5):
    dWx[:,H:2*H] = ((f_gate * (1 - f_gate) * dc * prev_c).T @ x).T
    dWh[:,H:2*H] = ((f_gate * (1 - f_gate) * dc * prev_c).T @ prev_h).T
    db[H:2*H] = (f_gate * (1 - f_gate) * dc * prev_c).sum(axis = 0)
    
    dx += (f_gate * (1 - f_gate) * dc * prev_c) @ Wx['f'].T
    dprev_h += (f_gate * (1 - f_gate) * dc * prev_c) @ Wh['f'].T
    #(2),(6):
    dWx[:,:H] = ((i_gate * (1 - i_gate) * dc * g_gate).T @ x).T
    dWh[:,:H] = ((i_gate * (1 - i_gate) * dc * g_gate).T @ prev_h).T
    db[:H] = (i_gate * (1 - i_gate) * dc * g_gate).sum(axis = 0)

    dx += (i_gate * (1 - i_gate) * dc * g_gate) @ Wx['i'].T
    dprev_h += (i_gate * (1 - i_gate) * dc * g_gate) @ Wh['i'].T
    #(2),(3):
    dWx[:,3*H:] = ((i_gate * (1 - g_gate**2) * dc).T @ x).T
    dWh[:,3*H:] = ((i_gate * (1 - g_gate**2) * dc).T @ prev_h).T
    db[3*H:] = (i_gate * (1 - g_gate**2) * dc).sum(axis = 0)

    dx += (i_gate * (1 - g_gate**2) * dc) @ Wx['g'].T
    dprev_h += (i_gate * (1 - g_gate**2) * dc) @ Wh['g'].T

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    N, T, D = x.shape
    H = Wx.shape[1]//4
    h = np.zeros((T+1, N, H))
    c = np.zeros((T+1, N, H))
    x = x.swapaxes(1, 0)
    cache = []
    h[0] = h0
    for i in range(T):
        h[i+1], c[i+1], tmp = lstm_step_forward(x[i], h[i], c[i], Wx, Wh, b)
        cache.append(tmp)
    h = h[1:,:,:].swapaxes(1, 0)
    cache.append(D)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    
    N, T, H = dh.shape
    D = cache[-1]
    dh = dh.swapaxes(1, 0)

    dx = np.zeros((T, N, D))
    dstep = np.zeros((T+1, N, H))
    dc = np.zeros((T+1, N, H))
    dWx = np.zeros((T, D, 4*H))
    dWh = np.zeros((T, H, 4*H))
    db = np.zeros((T, 4*H))
    for i in reversed(range(T)):
        dx[i], dstep[i], dc[i], dWx[i], dWh[i], db[i] =\
                            lstm_step_backward(dh[i]+dstep[i+1], dc[i+1], cache[i]) 
    dWx = dWx.sum(axis = 0)
    dWh = dWh.sum(axis = 0)
    dh0 = dstep[0]
    db = db.sum(axis = 0) 
    dx = dx.swapaxes(1, 0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
