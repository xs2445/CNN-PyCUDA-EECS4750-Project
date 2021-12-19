import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import time


class ConvLayerP:
    def __init__(self):
        # """
        # Attributes for instance of EncoderDecoder module
        # """
        self.mod = None
        self.path_kernel = "convkernel.cu"
        self.getSourceModule()
        self.TILE_WIDTH = 32

    def getSourceModule(self):
        """
        Get kernel from .cu file

        Args:
        - path: the path of the kernel.cu file
        """
        self.mod = SourceModule(open(self.path_kernel,"r").read())

    def forward_naive(self, X, Masks, N, C, M, H, W, K, format='NCHW', dtype=np.float32):
        """
        Naive parallel convolution without using shared or constant memory,
        the number and shape of threads blocks equals the shape of output matrix

        Properties:
        convolution layer:
        mode = valid
        stride = 1
        mask_width = K

        Parameters
        ----------
        X: input matrix with size [N, C, H, W] (NCHW format)
        Masks: masks with size [M, C, K, K] (NCHW format)
        N: number of samples 
        C: number of channels of input matrix
        M: number of channels of output matrix
        H: height of input matrix
        W: width of input matrix
        K: width of masks 
        format: channel_first (NCHW) or channel_last (NHWC) format. 
                channel_first format: X [N, C, H, W], Masks [M, C, K, K], Y [N, M, H-K+1, W-K+1].
                channel_last format: X [N, H, W, C], Masks [K, K, C, M], Y [N, H-K+1, W-K+1, M].

        Returns
        ----------
        Y: output matrix with size [N, M, H-K+1, W-K+1] (NCHW format)
        """

        start = cuda.Event()
        end = cuda.Event()
        start.record()

        X_d = gpuarray.to_gpu(X)
        Masks_d = gpuarray.to_gpu(Masks)
        w_y = W-K+1
        h_y = H-K+1

        if format == 'NCHW':
            Y_d = gpuarray.zeros((N, M, h_y,w_y), dtype=dtype)
            func = self.mod.get_function("convLayer_forward_naive")
        elif format == 'NHWC':
            Y_d = gpuarray.zeros((N, h_y, w_y, M), dtype=dtype)
            func = self.mod.get_function("convLayer_forward_naive_channel")
        else: 
            assert ValueError('The format should be NCHW or NHWC!')

        BlockDim = (self.TILE_WIDTH, self.TILE_WIDTH, 1)
        w_grid = w_y//self.TILE_WIDTH+1
        h_grid = h_y//self.TILE_WIDTH+1
        Num_tiles = w_grid * h_grid
        GridDim = (N, M, Num_tiles)

        func(X_d, Masks_d, Y_d, np.int32(N), np.int32(C), np.int32(M), np.int32(H), np.int32(W), np.int32(K), np.int32(w_grid), block=BlockDim, grid = GridDim)
        
        Y = Y_d.get()

        
        end.record()
        end.synchronize()

        return Y, start.time_till(end)


    def forward_shared(self, X, Masks, N, C, M, H, W, K, format='NCHW', dtype=np.float32):
        """
        Parallel convolution layer using shared memory,

        Properties:
        mode = valid
        stride = 1
        mask_width = K

        Parameters
        ----------
        X: input matrix with size [N, C, H, W]
        Masks: masks with size [M, C, K, K]
        N: number of samples 
        C: number of channels of input matrix
        M: number of channels of output matrix
        H: height of input matrix
        W: width of input matrix
        K: width of masks 
        format: choose channel_first (NCHW) or channel_last (NHWC) format. 
                channel_first format: X [N, C, H, W], Masks [M, C, K, K], Y [N, M, H-K+1, W-K+1].
                channel_last format: X [N, H, W, C], Masks [K, K, C, M], Y [N, H-K+1, W-K+1, M].
        dtype: the data type of X, Masks and Y

        Returns
        -------
        Y: output matrix with size [N, M, H-K+1, W-K+1]
        """

        start = cuda.Event()
        end = cuda.Event()
        start.record()

        # copy X and Masks to the device
        X_d = gpuarray.to_gpu(X)
        Masks_d = gpuarray.to_gpu(Masks)

        # outpout shape of Y
        w_y = W-K+1
        h_y = H-K+1
        
        # channel first or channel last format
        if format == 'NCHW':
            Y_d = gpuarray.zeros((N, M,h_y,w_y), dtype=dtype)
            func = self.mod.get_function("convLayer_forward_shared")
        elif format == 'NHWC':
            Y_d = gpuarray.zeros((N, h_y, w_y, M), dtype=dtype)
            func = self.mod.get_function("convLayer_forward_shared_channel")
        else: 
            assert ValueError('The format should be NCHW or NHWC!')
        
        # block size
        BlockDim = (self.TILE_WIDTH, self.TILE_WIDTH, 1)
        # grid size
        w_grid = w_y//self.TILE_WIDTH + 1
        h_grid = h_y//self.TILE_WIDTH + 1
        Num_tiles = w_grid * h_grid
        GridDim = (N, M, Num_tiles)
        # tile the input matrix to fit into blocks
        X_tile_width = self.TILE_WIDTH + K - 1
        # external defined shared memory size
        shm_space = (X_tile_width**2 + K**2) * np.dtype(dtype).itemsize
        
        func(
            X_d, 
            Masks_d, 
            Y_d, 
            np.int32(N), 
            np.int32(C), 
            np.int32(M), 
            np.int32(H), 
            np.int32(W), 
            np.int32(K), 
            np.int32(w_grid), 
            block=BlockDim, 
            grid = GridDim, 
            shared=shm_space)
        
        Y = Y_d.get()

        
        end.record()
        end.synchronize()

        return Y, start.time_till(end)


    def forward_sample_naive(self, X, Masks, C, M, H, W, K, format='CHW', dtype=np.float32):
        """
        Parallel convolution layer naive, only for 1 sample

        Properties:
        mode = valid,
        stride = 1,
        mask_width = K

        Parameters
        ----------
        X: input matrix with size [C, H, W]
        Masks: masks with size [M, C, K, K]
        C: number of channels of input matrix
        M: number of channels of output matrix
        H: height of input matrix
        W: width of input matrix
        K: width of masks 
        format: choose channel_first (NCHW) or channel_last (NHWC) format. 
                channel_first format: X [C, H, W], Masks [M, C, K, K], Y [M, H-K+1, W-K+1].
                channel_last format: X [H, W, C], Masks [K, K, C, M], Y [H-K+1, W-K+1, M].
        dtype: the data type of X, Masks and Y

        Returns
        -------
        Y: output matrix with size [M, H-K+1, W-K+1]
        """

        start = cuda.Event()
        end = cuda.Event()
        start.record()

        X_d = gpuarray.to_gpu(X)
        Masks_d = gpuarray.to_gpu(Masks)
        w_y = W-K+1
        h_y = H-K+1
        
        if format == 'CHW':
            Y_d = gpuarray.zeros((M,h_y,w_y), dtype=dtype)
            func = self.mod.get_function("convLayer_forward_sample_naive")
        elif format == 'HWC':
            Y_d = gpuarray.zeros((N, h_y, w_y, M), dtype=dtype)
            func = self.mod.get_function("convLayer_forward_sample_naive_channel")
        else: 
            assert ValueError('The format should be CHW or HWC!')
        
        BlockDim = (self.TILE_WIDTH, self.TILE_WIDTH, 1)
        GridDim = (w_y//self.TILE_WIDTH + 1, h_y//self.TILE_WIDTH + 1, M)
        
        func(
            X_d, 
            Masks_d, 
            Y_d, 
            np.int32(C), 
            np.int32(M), 
            np.int32(H), 
            np.int32(W), 
            np.int32(K), 
            block=BlockDim, 
            grid = GridDim)
        
        Y = Y_d.get()

        
        end.record()
        end.synchronize()

        return Y, start.time_till(end)


    def forward_sample_shared(self, X, Masks, C, M, H, W, K, format='CHW', dtype=np.float32):
        """
        Parallel convolution layer using shared memory, only for 1 sample

        Properties:
        mode = valid,
        stride = 1,
        mask_width = K

        Parameters
        ----------
        X: input matrix with size [C, H, W]
        Masks: masks with size [M, C, K, K]
        C: number of channels of input matrix
        M: number of channels of output matrix
        H: height of input matrix
        W: width of input matrix
        K: width of masks 
        format: choose channel_first (NCHW) or channel_last (NHWC) format. 
                channel_first format: X [C, H, W], Masks [M, C, K, K], Y [M, H-K+1, W-K+1].
                channel_last format: X [H, W, C], Masks [K, K, C, M], Y [H-K+1, W-K+1, M].
        dtype: the data type of X, Masks and Y

        Returns
        -------
        Y: output matrix with size [M, H-K+1, W-K+1]
        """

        start = cuda.Event()
        end = cuda.Event()
        start.record()

        X_d = gpuarray.to_gpu(X)
        Masks_d = gpuarray.to_gpu(Masks)
        w_y = W-K+1
        h_y = H-K+1
        
        if format == 'CHW':
            Y_d = gpuarray.zeros((M,h_y,w_y), dtype=dtype)
            func = self.mod.get_function("convLayer_forward_sample_shared")
        elif format == 'HWC':
            Y_d = gpuarray.zeros((N, h_y, w_y, M), dtype=dtype)
            func = self.mod.get_function("convLayer_forward_sample_shared_channel")
        else: 
            assert ValueError('The format should be CHW or HWC!')
        
        # block size
        BlockDim = (self.TILE_WIDTH, self.TILE_WIDTH, 1)
        # grid size
        GridDim = (w_y//self.TILE_WIDTH + 1, h_y//self.TILE_WIDTH + 1, M)

        # tile the input matrix to fit into blocks
        X_tile_width = self.TILE_WIDTH + K - 1
        # external defined shared memory size
        shm_space = (X_tile_width**2 + K**2) * np.dtype(dtype).itemsize
        
        func(
            X_d, 
            Masks_d, 
            Y_d, 
            np.int32(C), 
            np.int32(M), 
            np.int32(H), 
            np.int32(W), 
            np.int32(K), 
            block=BlockDim, 
            grid = GridDim,
            shared=shm_space)
        
        Y = Y_d.get()
        
        end.record()
        end.synchronize()

        return Y, start.time_till(end)


    def forward_serial(self, X, Masks, N, C, M, H, W, K, format='NCHW', dtype=np.float32):
        """
        Serial forward convolutional computation

        Properties:
        mode = valid,
        stride = 1,
        mask_width = K

        Parameters
        ----------
        X: input matrix with size [N, C, H, W]
        Masks: masks with size [M, C, K, K]
        N: number of samples 
        C: number of channels of input matrix
        M: number of channels of output matrix
        H: height of input matrix
        W: width of input matrix
        K: width of masks 
        format: choose channel_first (NCHW) or channel_last (NHWC) format. 
                channel_first format: X [N, C, H, W], Masks [M, C, K, K], Y [N, M, H-K+1, W-K+1].
                channel_last format: X [N, H, W, C], Masks [K, K, C, M], Y [N, H-K+1, W-K+1, M].
        dtype: the data type of X, Masks and Y

        Returns
        -------
        Y: output matrix with size [N, M, H-K+1, W-K+1]
        """

        start = time.time()

        out_height = H-K+1
        out_width = W-K+1

        if format == 'NCHW':
            # create result matrix
            out = np.zeros((N, M, out_height, out_width)).astype(dtype)
            
            for bat in range(N):
                for row_out in range(out_height):
                    for col_out in range(out_width):
                        for cha_out in range(M):
                            result = 0
                            for cha_in in range(C):
                                # the slice of x for convolution
                                x_slice = X[bat, cha_in, row_out:row_out+K, col_out:col_out+K]
                                # the kernel (here the convolution operation is actually correlation, 
                                # then the BP will use convolution to implement)
                                kernel_slice = Masks[cha_out,cha_in,:,:]
                                result += np.sum(x_slice * kernel_slice)
                            # result
                            out[bat,cha_out,row_out,col_out] = result

        elif format == 'NHWC':
            # create result matrix
            out = np.zeros((N, out_height, out_width, M)).astype(dtype)
            
            for bat in range(N):
                for row_out in range(out_height):
                    for col_out in range(out_width):
                        for cha_out in range(M):
                            result = 0
                            for cha_in in range(C):
                                # the slice of x for convolution
                                x_slice = X[bat, row_out:row_out+K, col_out:col_out+K, cha_in]
                                # the kernel (here the convolution operation is actually correlation, 
                                # then the BP will use convolution to implement)
                                kernel_slice = Masks[:,:, cha_in, cha_out]
                                result += np.sum(x_slice * kernel_slice)
                            # result
                            out[bat,row_out,col_out,cha_out] = result

        else: 
            assert ValueError('The format should be NCHW or NHWC!')   
            
        return out, (time.time()-start)*1e3