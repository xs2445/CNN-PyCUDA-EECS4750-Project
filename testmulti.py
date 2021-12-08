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
        //---------naive ends-------------------------------------------
        
        
        //---------shared starts----------------------------------------
        #define blocksize 32  // which equals tile size and blockDim.x,y
                                // should be the same in host code
                                
        __global__ void mul_shared(float *M, float *N, 
                                  const int Mw, const int Mh, const int Nw, 
                                  float *P){
        // input M size = Mh*Mw
        // input N size = Mw*Nw
        // output P size = Mh*Nw
            
            __shared__ float MS[blocksize][blocksize];
            __shared__ float NS[blocksize][blocksize];
            
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int bx = blockIdx.x;
            int by = blockIdx.y;
            
            int row = ty + by*blocksize;
            int col = tx + bx*blocksize;
            
            // int strideM = ceil(Mh/blocksize);
            int strideM = Mw/blocksize +1;
            
            float temp = 0.0f;
            for(int i=0; i<strideM;i++){
                
                if (i*blocksize+tx<Mw && row <Mh){
                    MS[ty][tx] = M[row*Mw+i*blocksize+tx];
                }
                else{
                    MS[ty][tx] = 0.0f;       //padding block elementsthat exceeds M's margin with 0
                }
                
                if (i*blocksize+ty<Mw && col <Nw){
                    NS[ty][tx] = N[(i*blocksize+ty)*Nw+col];
                }
                else{
                    NS[ty][tx] = 0.0f;
                }
                __syncthreads();
                
                
                for(int j=0;j<blocksize;j++){
                    temp += MS[ty][j]*NS[j][tx];
                }
                __syncthreads();
            }
            if(row<Mh && col< Nw){
                P[row*Nw+col] = temp;
            }
        }
        //----------shared ends-----------------------------------------------
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

    def multi_tile(self, M, N, M_w, M_h, N_w):
        P = np.zeros(shape=(M_h, N_w)).astype(np.float32)  # create output array in host
        func = self.mod.get_function("mul_shared")
        start = cuda.Event()
        end = cuda.Event()
        start.record()
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


def pytest(a, b):
    start = time.time()
    c = np.dot(a, b)
    end = time.time()
    return c, (end-start)*1e3


block = 32
blocksize = (block, block, 1)

if __name__ == "__main__":
    cuda_model = multi()

    M_h = 2048/2
    M_w = 2048/2
    N_w = 2048/2
    M_h = np.int32(M_h)
    M_w = np.int32(M_w)
    N_w = np.int32(N_w)

    M = np.random.randint(1, 5, (M_h, M_w)).astype(np.float32)
    N = np.random.randint(1, 5, (M_w, N_w)).astype(np.float32)
    P_naive,t_naive = cuda_model.multi_naive(M,N,M_w,M_h,N_w)
    P_shared,t_shared = cuda_model.multi_tile(M,N,M_w,M_h,N_w)
    P_py,t_py = pytest(M, N)
    try:
        # print("Checkpoint: Do python and gpu convolution match? Checking...")
        assert ((P_shared == P_py).all())
        print("match, result:")
    except AssertionError:
        print("conv results do not match ")

    print("python test = \n",P_py)
    print("P_shared = \n",P_shared)

    print("t_naive=\n", t2-t1)
    print("t_shared=\n",t3-t2)
    print("t_py=\n",t4-t3)
