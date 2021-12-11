import numpy as np
# from Layer import Layer



class Dense(object):
    def __init__(self, size_in, size_out, dtype=np.float32):
        """
        Fully Connected layer X*W+B
        The input shape could be [batch, d1, ..., dn]
        This layer will flatten the input matrix to [batch, -1]

        Args:
        - size_in: the input shape could be [d1, ..., dn]
        - size_out: the shape of the output [size_out]
        """
        # super().__init__(size_in=size_in, dtype=dtype)
        self.layer_name = 'Dense'
        # input shape
        self.size_in = size_in
        # flattened length of the matrix
        
        # if len(size_out):
        #     result = 1
        #     for x in size_in:
        #         result *= x
        #     self.length_in = result
        # else:
        #     self.length = size_out
        self.length_in = size_in

        # output shape
        self.size_out = size_out
        # self.params = []
        self.params = dict()

        # initialize the weight with size (length_in, size_out)
        self.params["weight"] = np.random.rand(self.length_in, self.size_out).astype(dtype)
        # self.params.append(np.random.rand(self.length_in, self.size_out).astype(dtype))
        
        # initialize the bias with size (size_out)
        self.params["bias"]= np.random.rand(self.size_out).astype(dtype)
        # self.params.append(np.random.rand(self.size_out).astype(dtype))

        # initialize the grad
        self.grads= dict()

        # cache the input matrix
        self.x = None


    def forward(self, x):
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

        self.x = x.copy()

        w = self.params["weight"]
        b = self.params["bias"]

        batch = x.shape[0]
        # flatten the input matrix
        x_flatten = x.reshape((batch, -1))
        out = np.dot(x_flatten, w) + b

        return out


        
    def backward(self, dout):
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
        x = self.x
        w = self.params["weight"]
        # b = self.b

        N = x.shape[0]
        x_flatten = x.reshape((N, -1))

        dx = np.reshape(np.dot(dout, w.T), x.shape)
        dw = np.dot(x_flatten.T, dout)
        db = np.dot(np.ones((N,)), dout)

        self.grads["weight"] = dw
        self.grads["bias"] = db

        return dx


        


        



