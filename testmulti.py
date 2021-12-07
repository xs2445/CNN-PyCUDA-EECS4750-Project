import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
import matplotlib.pyplot as plt


class multi:

    def __init__(self):
        self.mod = self.getSourceModule()

    def getgriddim(self, a, b):
        a = int(a)
        b = int(b)
        N = a // b + 1
        return N

    def getSourceModule(self):
        kernelwrapper = """
        __global__ void mul_naive(float *M, float *N, 
                                  const int M_w, const int M_h, const int N_w, 
                                  float *P){
        // input M size = M_h*M_w
        // input N size = M_w*N_w
        // output P size = M_h*N_w
            int row = threadIdx.y + blockIdx.y*blockDim.y;
            int col = threadIdx.x + blockIdx.x*blockDim.x;
            
            if ((row<M_h)&&(col<N_w)){
            float temp = 0.0f;
            
            for(int i=0; i<M_w;i++){
                temp += M[i+row*M_w]*N[col+i*N_w];
            }
            
            P[row*M_w+col] += temp;
        }
        }
        
        """
        return SourceModule(kernelwrapper)

    def multi_naive(self, M, N, M_w, M_h, N_w):
        # input M size = M_h * M_w
        # input N size = M_w * N_w
        # output P size = M_h * N_w
        P = np.zeros(shape=(M_h, N_w)).astype(np.float32)  # create output array in host
        func = self.mod.get_function("mul_naive")
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        P_size = M_h * N_w
        dM = cuda.mem_alloc_like(M)  # create input array in device
        dN = cuda.mem_alloc_like(N)
        dP = cuda.mem_alloc_like(P)
        cuda.memcpy_htod(dM, M)  # copy input array from host to device
        cuda.memcpy_htod(dN, N)


        # kernel call
        griddimx = self.getgriddim(M_h, block)
        griddimy = self.getgriddim(N_w, block)
        func(dM, dN, M_w, M_h, N_w, dP, block=blocksize, grid=(griddimx, griddimy, 1))
        cuda.memcpy_dtoh(P, dP)
        end.record()
        end.synchronize()

        return P, start.time_till(end)

    def multi_tile(self):
        pass


def pytest(a, b):
    start = time.time()
    c = np.dot(a, b)
    end = time.time()
    return c, (end-start)*1e3


block = 32
blocksize = (block, block, 1)

if __name__ == "__main__":
    cuda_model = multi()

    M_h = 1024
    M_w = 1024
    N_w = 1024
    M_h = np.int32(M_h)
    M_w = np.int32(M_w)
    N_w = np.int32(N_w)

    M = np.random.randint(1, 5, (M_h, M_w)).astype(np.float32)
    N = np.random.randint(1, 5, (M_w, N_w)).astype(np.float32)

    P_naive,t_naive = cuda_model.multi_naive(M,N,M_w,M_h,N_w)
    P_py,t_py = pytest(M, N)

    try:
        # print("Checkpoint: Do python and gpu convolution match? Checking...")
        assert ((P_naive == P_py).all())
        print("match, result:")
    except AssertionError:
        print("conv results do not match ")

    print("python test = \n",P_py)
    print("P_naive = \n",P_naive)
    print("t_naive=\n",t_naive)
    print("t_py=\n",t_py)