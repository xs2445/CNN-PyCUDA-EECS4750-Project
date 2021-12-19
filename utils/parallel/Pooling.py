import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
import matplotlib.pyplot as plt


class pooling:

    def __init__(self):
        self.mod = self.getSourceModule()
        self.blocksize = 16
        self.path_kernel = "pooling.cu"

    def getgriddim(self, a, b):
        a = int(a)
        b = int(b)
        N = a // b + 1
        return N

    def getSourceModule(self):
        return SourceModule(open(self.path_kernel,"r").read())

    def ave_naive(self, M, stride, window):
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        blocksize = self.blocksize
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


