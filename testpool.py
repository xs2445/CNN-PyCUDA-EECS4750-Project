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
        
        __device__ int global_id_3d(int n1, int n2, int n3, int N2, int N3){
            //return n3 + N3*global_id_2d(n1,n2,N2);
            return n3 + N3*(n2 + N2*n1);
        }
        
        
        
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
        
        __global__ void pool_ave_3d(float* M, float* P, 
                                       const int wM, const int hM,
                                       const int stride, const int window,
                                       const int wP, const int hP){
          // window must have the same width and height

            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int channel = blockIdx.z;
            int idx = tx+blockIdx.x*blockDim.x;
            int idy = ty+blockIdx.y*blockDim.y;
            
            float temp = 0.0f;
            
            for(int i=0;i<window;i++){   // height direction
                for(int j=0;j<window;j++){  // width direction
                    temp += M[global_id_3d(channel,idy*stride,idx*stride,hM,wM)+i*wM+j];  
                }
            }
            __syncthreads();
            
            temp = temp/(window*window); // get mean
            
            if(idx<wP && idy<hP){
                P[global_id_3d(channel,idy,idx,hP,wP)] += temp;
            }
        }   
        
        """
        return SourceModule(kernelwrapper)

    # used for 2D matrix pooling testing. no longer used in this py
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
        # print("P.shape", P.shape)
        dM = cuda.mem_alloc_like(M)
        dP = cuda.mem_alloc_like(P)

        cuda.memcpy_htod(dM, M)
        block = (blocksize, blocksize, 1)
        grid = (self.getgriddim(hP - 1, blocksize), self.getgriddim(wP - 1, blocksize), 1)
        # print("grid=", grid)
        func(dM, dP, wM, hM, stride, window, wP, hP, block=block, grid=grid)
        cuda.memcpy_dtoh(P, dP)
        end.record()
        end.synchronize()
        return P, start.time_till(end)

    def pooling_py(self, M, stride, window):
        start = time.time()
        channel, hM, wM = M.shape
        hP = self.getgriddim(hM - window, stride)
        wP = self.getgriddim(wM - window, stride)
        P = np.zeros(shape=(channel, hP, wP))
        for k in range(channel):
            for i in range(hP):
                for j in range(wP):
                    # print("i=", i)
                    # print("j=", j)
                    P[k, i, j] = np.mean(M[k, i * stride:i * stride + window, j * stride:j * stride + window])
                    # print(M[i * stride:i * stride + window, j * stride:j * stride + window])

        end = time.time()
        return P, (end - start) * 1e3

    def ave_naive_3d(self, M, stride, window):
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        func = self.mod.get_function("pool_ave_3d")
        channel, hM, wM = M.shape
        hM = np.int32(hM)
        wM = np.int32(wM)
        # zM = np.int32(channel)

        hP = np.int32(self.getgriddim(hM - window, stride))
        wP = np.int32(self.getgriddim(wM - window, stride))
        P = np.zeros(shape=(channel, hP, wP)).astype(np.float32)
        print("P.shape", P.shape)
        dM = cuda.mem_alloc_like(M)
        dP = cuda.mem_alloc_like(P)

        cuda.memcpy_htod(dM, M)
        block = (blocksize, blocksize, 1)
        grid = (self.getgriddim(hP - 1, blocksize), self.getgriddim(wP - 1, blocksize), channel)
        print("grid=", grid)
        func(dM, dP, wM, hM, stride, window, wP, hP, block=block, grid=grid)
        cuda.memcpy_dtoh(P, dP)
        end.record()
        end.synchronize()
        return P, start.time_till(end)


if __name__ == "__main__":
    cuda_model = pooling()
    blocksize = 32
    # wM = 8
    # hM = 8
    t_py_list = []
    t_naive_list = []
    t_single_list = []
    N = 5
    size = [32,128,512,1024]
    stride = np.int32(2)
    window_size = np.int32(2)
    # 3 channel matrix
    # do not consider batch yet
    for i in size:
        M_1 = np.random.randint(1, 3, (N, i, i)).astype(np.float32)
        M_2 = np.random.randint(1, 3, (N, i, i)).astype(np.float32)
        M_3 = np.random.randint(1, 3, (N, i, i)).astype(np.float32)
        # print(M_1)
        P_py, t_py = cuda_model.pooling_py(M_1, stride, window_size)
        P_naive, t_naive = cuda_model.ave_naive_3d(M_1, stride, window_size)

        t_single = 0
        P_2D = np.zeros(shape=(N, int(i/2), int(i/2)))
        for k in range(N):
            P_2D[k, :], t_single_i = cuda_model.ave_naive(M_1[k, :], stride, window_size)
            t_single += t_single_i
        try:
            # print("Checkpoint: Do python and gpu convolution match? Checking...")
            assert ((P_2D == P_py).all())
            print("match")
        except AssertionError:
            print("pool results do not match ")
        t_py_list.append(t_py)
        t_naive_list.append(t_naive)
        t_single_list.append(t_single)

        # print('M1\n', M_1)
        # print('py\n', P_py)
        # print('3d\n', P_naive)
    fig1 = plt.figure()
    plt.title("PyCUDA ave_pool")
    plt.xlabel('size of array')
    plt.ylabel('average execution time/ms')
    plt.plot(size, t_naive_list, 'b', label="naive pool")

    # plt.plot(size, t_py_list, 'r', label="numpy.mean")
    plt.plot(size, t_single_list, 'r', label="2D iteration")
    plt.legend()
    plt.show()
    fig1.savefig('pycuda_pool.png')
