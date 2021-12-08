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
            int strideM = Mw/blocksize+1;
            
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
        griddimx = self.getgriddim(N_w, block)
        griddimy = self.getgriddim(M_h, block)
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
        griddimx = self.getgriddim(N_w, block)
        griddimy = self.getgriddim(M_h, block)
        func(dM, dN, M_w, M_h, N_w, dP, block=blocksize, grid=(griddimx, griddimy, 1))
        cuda.memcpy_dtoh(P, dP)
        end.record()
        end.synchronize()
        return P, start.time_till(end)


def pytest(a, b):
    start = time.perf_counter()
    c = np.dot(a, b)
    end = time.perf_counter()
    return c, (end - start) * 1e3


block = 32
blocksize = (block, block, 1)

if __name__ == "__main__":
    cuda_model = multi()
    t_naive_list = []
    t_shared_list = []
    t_py_list = []
    size = [1024, 2048, 4096]

    for i in size:
        M_h = i
        M_w = i * 3
        N_w = i * 8
        M_h = np.int32(M_h)
        M_w = np.int32(M_w)
        N_w = np.int32(N_w)

        M = np.random.randint(1, 5, (M_h, M_w)).astype(np.float32)
        N = np.random.randint(1, 5, (M_w, N_w)).astype(np.float32)
        P_naive, t_naive = cuda_model.multi_naive(M, N, M_w, M_h, N_w)
        P_shared, t_shared = cuda_model.multi_tile(M, N, M_w, M_h, N_w)
        P_py, t_py = pytest(M, N)
        try:
            # print("Checkpoint: Do python and gpu convolution match? Checking...")
            assert ((P_shared == P_py).all())
            print("match, result:")
        except AssertionError:
            print("conv results do not match ")
        t_naive_list.append(t_naive)
        t_shared_list.append(t_shared)
        t_py_list.append(t_py)
        print("t_naive=\n", t_naive)
        print("t_shared=\n", t_shared)
        print("t_py=\n", t_py)
    fig1 = plt.figure()
    plt.title("PyCUDA multiplication")
    plt.xlabel('size of array')
    plt.ylabel('average execution time/ms')
    plt.plot(size, t_naive_list, 'b', label="naive ")
    plt.plot(size, t_shared_list, 'g', label="shared memory")
    plt.plot(size, t_py_list, 'r', label="numpy.dot")
    plt.legend()
    plt.show()
    fig1.savefig('pycuda_conv.png')
