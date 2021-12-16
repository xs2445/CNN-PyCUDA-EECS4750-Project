import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
import matplotlib.pyplot as plt


class pooling:

    def __init__(self):
        self.mod = self.getSourceModule()

    def getgriddim(self, a, b):
        a = int(a)
        b = int(b)
        N = a // b + 1
        return N

    def getSourceModule(self):
        kernelwrapper = """
        __global__ void pool_ave_naive(float* M, float* P, 
                                       const int wM, const int hM,
                                       const int stride, const int window,
                                       const int wP, const int hP){
          // window must have the same width and height

            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int idx = tx+blockIdx.x*blockDim.x;
            int idy = ty+blockIdx.y*blockDim.y;
            
            float temp = 0.0f;
            
            for(int i=0;i<window;i++){
                for(int j=0;j<window;j++){
                    temp += M[(idy*stride+i)*wM+idx*stride+j];  
                }
            }
            __syncthreads();
            
            temp = temp/(window*window); // get mean
            if(idx<wP && idy<hP){
                P[(idy)*wP+idx] = temp;
            }
        }   
        """
        return SourceModule(kernelwrapper)

    def ave_naive(self, M, stride, window):
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        func = self.mod.get_function("pool_ave_naive")
        hM, wM = M.shape
        hM = np.int32(hM)
        wM = np.int32(wM)

        hP = np.int32(self.getgriddim(hM - window, stride))
        wP = np.int32(self.getgriddim(wM - window, stride))
        P = np.zeros(shape=(hP, wP)).astype(np.float32)
        print("P.shape",P.shape)
        dM = cuda.mem_alloc_like(M)
        dP = cuda.mem_alloc_like(P)

        cuda.memcpy_htod(dM, M)
        block = (blocksize, blocksize, 1)
        grid = (self.getgriddim(hP-1, blocksize), self.getgriddim(wP-1, blocksize), 1)
        print("grid=",grid)
        func(dM, dP, wM, hM, stride, window,wP,hP,block=block, grid=grid)
        cuda.memcpy_dtoh(P, dP)
        end.record()
        end.synchronize()
        return P, start.time_till(end)

    def pooling_py(self, M, stride, window):
        start = time.time()
        hM, wM = M.shape
        hP = self.getgriddim(hM - window, stride)
        wP = self.getgriddim(wM - window, stride)
        P = np.zeros(shape=(hP, wP))

        for i in range(hP):
            for j in range(wP):
                # print("i=", i)
                # print("j=", j)
                P[i, j] = np.mean(M[i * stride:i * stride + window, j * stride:j * stride + window])
                # print(M[i * stride:i * stride + window, j * stride:j * stride + window])

        end = time.time()
        return P, (end - start) * 1e3


if __name__ == "__main__":
    cuda_model = pooling()
    blocksize = 16
    # wM = 8
    # hM = 8

    size = [16,128,512,1024,2048]
    stride = np.int32(2)
    window_size = np.int32(2)
    # 3 channel matrix
    # do not consider batch yet
    for i in size:

        M_1 = np.random.randint(1, 3, (size, size)).astype(np.float32)
        M_2 = np.random.randint(1, 3, (size, size)).astype(np.float32)
        M_3 = np.random.randint(1, 3, (size, size)).astype(np.float32)

        # print(M_1)
        P_py, t_py = cuda_model.pooling_py(M_1, stride, window_size)
        P_naive, t_naive = cuda_model.ave_naive(M_1, stride, window_size)

        print(M_1)
        print(P_py)
        print(P_naive)
