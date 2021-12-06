import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)

    :return:
    - out: output, of shape (N, M)
    - cache: x, w, b for back-propagation
    """
    num_train = x.shape[0]
    x_flatten = x.reshape((num_train, -1))
    out = np.dot(x_flatten, w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
                    x: Input data, of shape (N, d_1, ... d_k)
                    w: Weights, of shape (D, M)

    :return: a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache

    N = x.shape[0]
    x_flatten = x.reshape((N, -1))

    dx = np.reshape(np.dot(dout, w.T), x.shape)
    dw = np.dot(x_flatten.T, dout)
    db = np.dot(np.ones((N,)), dout)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    :param x: Inputs, of any shape
    :return: A tuple of:
    - out: Output, of the same shape as x
    - cache: x for back-propagation
    """
    out = np.zeros_like(x)
    out[np.where(x > 0)] = x[np.where(x > 0)]

    cache = x

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    :param dout: Upstream derivatives, of any shape
    :param cache: Input x, of same shape as dout

    :return: dx - Gradient with respect to x
    """
    x = cache

    dx = np.zeros_like(x)
    dx[np.where(x > 0)] = dout[np.where(x > 0)]

    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    y_prediction = argmax(softmax(x))

    :param x: (float) a tensor of shape (N, #classes)
    :param y: (int) ground truth label, a array of length N

    :return: loss - the loss function
             dx - the gradient wrt x
    """
    loss = 0.0
    num_train = x.shape[0]

    x = x - np.max(x, axis=1, keepdims=True)
    x_exp = np.exp(x)
    loss -= np.sum(x[range(num_train), y])
    loss += np.sum(np.log(np.sum(x_exp, axis=1)))

    loss /= num_train

    neg = np.zeros_like(x)
    neg[range(num_train), y] = -1

    pos = (x_exp.T / np.sum(x_exp, axis=1)).T

    dx = (neg + pos) / num_train

    return loss, dx


def conv2d_forward(x, w, b, pad, stride):
    """
    A Numpy implementation of 2-D image convolution.
    By 'convolution', simple element-wise multiplication and summation will suffice.
    The border mode is 'valid' - Your convolution only happens when your input and your filter fully overlap.
    Another thing to remember is that in TensorFlow, 'padding' means border mode (VALID or SAME). For this practice,
    'pad' means the number rows/columns of zeroes to concatenate before/after the edge of input.

    Inputs:
    :param x: Input data. Should have size (batch, height, width, channels).
    :param w: Filter. Should have size (filter_height, filter_width, channels, num_of_filters).
    :param b: Bias term. Should have size (num_of_filters, ).
    :param pad: Integer. The number of zeroes to pad along the height and width axis.
    :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

    :return: A 4-D array. Should have size (batch, new_height, new_width, num_of_filters).

    Note:
    To calculate the output shape of your convolution, you need the following equations:
    new_height = ((height - filter_height + 2 * pad) // stride) + 1
    new_width = ((width - filter_width + 2 * pad) // stride) + 1
    For reference, visit this website:
    https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/
    """

    # size of input matrix
    batch, height, width, channels = x.shape
    
    # size of kernel
    filter_height, filter_width, filter_channel, num_filter = w.shape
        
    # the size of the output matrix in mode "same"
    out_height = (height - filter_height + 2*pad)//stride + 1
    out_width = (width - filter_width + 2*pad)//stride + 1
    # create result matrix
    out = np.zeros((batch, out_height, out_width, num_filter))
    
    for bat in range(batch):
        x_padding = np.pad(x[bat,:,:,:], ((pad,pad),(pad,pad),(0,0)), 'constant', constant_values=0)
        for row_out in range(out_height):
            row_start = row_out * stride
            for col_out in range(out_width):
                col_start = col_out * stride
                for cha_out in range(num_filter):
                    result = 0
                    for cha_in in range(channels):
                        # the slice of x for convolution
                        x_slice = x_padding[row_start:row_start+filter_height, col_start:col_start+filter_width, cha_in]
                        # the kernel (here the convolution operation is actually correlation, 
                        # then the BP will use convolution to implement)
                        kernel_slice = w[:,:,cha_in,cha_out]
                        result += np.sum(x_slice * kernel_slice)
                    # result
                    out[bat,row_out,col_out,cha_out] = result + b[cha_out]
    
    return out


def conv2d_backward(d_top, x, w, b, pad, stride):
    """
    (Optional, but if you solve it correctly, we give you 5 points for this assignment.)
    A lite Numpy implementation of 2-D image convolution back-propagation.

    Inputs:
    :param d_top: The derivatives of pre-activation values from the previous layer
                       with shape (batch, height_new, width_new, num_of_filters).
    :param x: Input data. Should have size (batch, height, width, channels).
    :param w: Filter. Should have size (filter_height, filter_width, channels, num_of_filters).
    :param b: Bias term. Should have size (num_of_filters, ).
    :param pad: Integer. The number of zeroes to pad along the height and width axis.
    :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

    :return: (d_w, d_b), i.e. the derivative with respect to w and b. For example, d_w means how a change of each value
     of weight w would affect the final loss function.

    Note:
    Normally we also need to compute d_x in order to pass the gradients down to lower layers, so this is merely a
    simplified version where we don't need to back-propagate.
    For reference, visit this website:
    http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
    """
    
    # for derivative of dx, dx = dout * w (mode='same', here '*' represents convolution)
    
    # for derivative of dw, dw = x * dout (mode='valid', here '*' represents correlation)
    # size of input matrix
    batch, height, width, channels = x.shape
    
    # size of kernel
    filter_height, filter_width, filter_channel, num_filter = w.shape
    
    _, out_height, out_width, _ = d_top.shape
    
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    for cha_in in range(channels):
        for cha_out in range(num_filter):
            for bat in range(batch):
                # convolution of x and dout
                for row_out in range(filter_height):
                    row_start = row_out * stride
                    for col_out in range(filter_width):
                        col_start = col_out * stride
                        x_slice = x[bat, row_start:row_start+out_height, col_start:col_start+out_width, cha_in]
                        dtop_slice = d_top[bat,:,:,cha_out]
                        dw[row_out,col_out,cha_in,cha_out] += np.sum(x_slice * dtop_slice)
                    
    dw /= batch
    
    for cha_out in range(num_filter):
        db[cha_out] = np.sum(d_top[:,:,:,cha_out])
    # db /= batch
        
    return dw, db, dw.shape


def avg_pool_forward(x, pool_size, stride):
    """
    A Numpy implementation of 2-D image average pooling.

    Inputs:
    :params x: Input data. Should have size (batch, height, width, channels).
    :params pool_size: Integer. The size of a window in which you will perform average operations.
    :params stride: Integer. The number of pixels to move between 2 neighboring receptive fields.
    :return :A 4-D array. Should have size (batch, new_height, new_width, num_of_filters).
    """
    
    # size of input matrix
    batch, height, width, channels = x.shape
            
    # the size of the output matrix in mode "same"
    out_height = (height - pool_size)//stride + 1
    out_width = (width - pool_size)//stride + 1
    # create result matrix
    out = np.zeros((batch, out_height, out_width, channels))
    
    for bat in range(batch):
        x_padding = x[bat,:,:,:]
        for row_out in range(out_height):
            row_start = row_out * stride
            for col_out in range(out_width):
                col_start = col_out * stride
                for cha_in in range(channels):
                    # the slice of x for convolution
                    x_slice = x_padding[row_start:row_start+pool_size, col_start:col_start+pool_size, cha_in]
                    # result
                    out[bat,row_out,col_out,cha_in] = np.mean(x_slice)
    
    return out

    
def avg_pool_backward(dout, x, pool_size, stride):
    """
    (Optional, but if you solve it correctly, we give you +5 points for this assignment.)
    A Numpy implementation of 2-D image average pooling back-propagation.

    Inputs:
    :params dout: The derivatives of values from the previous layer
                       with shape (batch, height_new, width_new, num_of_filters).
    :params x: Input data. Should have size (batch, height, width, channels).
    :params pool_size: Integer. The size of a window in which you will perform average operations.
    :params stride: Integer. The number of pixels to move between 2 neighboring receptive fields.
    
    :return dx: The derivative with respect to x
    You may find this website helpful:
    https://medium.com/the-bioinformatics-press/only-numpy-understanding-back-propagation-for-max-pooling-layer-in-multi-layer-cnn-with-example-f7be891ee4b4
    """
    
    # size of input matrix
    batch, height, width, channels = x.shape
    
    _, out_height, out_width, _ = dout.shape
    
    dx = np.zeros_like(x)
    
    for bat in range(batch):
        for row_out in range(out_height):
            row_start = row_out * stride
            for col_out in range(out_width):
                col_start = col_out * stride
                for cha_in in range(channels):
                    # upsampling
                    dx[bat, row_start:row_start+pool_size, col_start:col_start+pool_size, cha_in] = dout[bat, row_out,col_out, cha_in]/(pool_size**2)
                    
    return dx
