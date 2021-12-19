import numpy as np
from Layer import Layer
from layer_functions import conv2d_forward, conv2d_backward



class Conv2d(Layer):
    def __init__(self, channels_out, mask_size, size_in, dtype=np.float32, stride=1, pad=0):
        """
        Convolutional layer
        The size of the mask is (mask_size, mask_size, channels_in, channels_out)


        Args:
        - channels_out: the number of filters
        - mask_size: the size of the mask, should be a single int value
        - size_in: the size of input matrix with shape (height, width, channels_in)
        """
        # super().__init__(size_in=size_in, dtype=dtype)

        # the size of masks
        self.mask_size = (mask_size, mask_size, size_in[-1], channels_out)
        # the size of the output matrix in mode "same"
        out_height = (size_in[0] - mask_size + 2*pad)//stride + 1
        out_width = (size_in[1] - mask_size + 2*pad)//stride + 1
        # the size of input matrix
        self.size_in = size_in
        # the size of result (out_height, out_width, channels_out)
        self.size_out = (out_height, out_width, channels_out)
        # the number of input channel
        self.channels_in = size_in[2]
        # the number of output channel
        self.channels_out = channels_out
        # width of padding
        self.pad = pad
        # stride
        self.stride = stride

        # initialize the masks
        self.masks = np.random.rand(mask_size, dtype=dtype)
        # initialize the biases
        self.bias = np.random.rand(channels_out, dtype=dtype)
        # cache the input matrix
        self.cache = None


    def forward(self, x):
        """
        Forward convolutional computation 

        Args:
        - x: the imput matrix with shape (batch, width, length, channels_in)

        """
        batch = x.shape[0]
        self.cache = x.copy()

        channels = self.channels_in
        out_height, out_width, num_filter = self.size_out
        filter_height, filter_width = self.mask_size
        pad = self.pad
        stride = self.stride
        w = self.masks
        b = self.bias


        
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

    def backword(self, d_top):
        pass
        



