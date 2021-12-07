import numpy as np
from Layer import Layer
from layer_functions import conv2d_forward, conv2d_backward



class AveragePooling(Layer):
    def __init__(self, pool_size, size_in, stride=2):
        """
        Average pooling layer
        The output shape should be (height_out, width_out, channels_in)

        Args:
        - pool_size: the size of pooling mask
        - size_in: the size of input matrix with shape (height, width, channels_in)
        - stride
        """
        # super().__init__(size_in=size_in, dtype=dtype)

        # the size of masks
        self.pool_size = pool_size
        self.stride = stride
        self.size_in = size_in
        height_out = (size_in[0]-pool_size)//stride
        width_out = (size_in[1]-pool_size)//stride
        self.size_out = (height_out, width_out, size_in[2])
        self.x = None


    def forward(self, x):
        """
        A Numpy implementation of 2-D image average pooling.

        Inputs:
        :params x: Input data. Should have size (batch, height, width, channels).
        :params pool_size: Integer. The size of a window in which you will perform average operations.
        :params stride: Integer. The number of pixels to move between 2 neighboring receptive fields.
        :return :A 4-D array. Should have size (batch, new_height, new_width, num_of_filters).
        """

        self.x = x.copy()

        stride = self.stride
        pool_size = self.pool_size
        
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


    def backword(self, dout):
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
        batch = self.x.shape[0]

        out_height, out_width, channels = self.size_out

        pool_size = self.pool_size
        stride = self.stride
        
        dx = np.zeros_like((batch, self.size_in))
        
        for bat in range(batch):
            for row_out in range(out_height):
                row_start = row_out * stride
                for col_out in range(out_width):
                    col_start = col_out * stride
                    for cha_in in range(channels):
                        # upsampling
                        dx[bat, row_start:row_start+pool_size, col_start:col_start+pool_size, cha_in] = dout[bat, row_out,col_out, cha_in]/(pool_size**2)
                        
        return dx



