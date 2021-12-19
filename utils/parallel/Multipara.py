import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
import matplotlib.pyplot as plt


class multi:

    def __init__(self):
        self.mod = self.getSourceModule()
        self.block = 32
        self.path_kernel = "multiplication.cu"

    def getgriddim(self, a, b):
        a = int(a)
        b = int(b)
        N = a // b + 1
        return N

    def getSourceModule(self):
        return SourceModule(open(self.path_kernel,"r").read())

    def multi_naive(self, M, N, M_w, M_h, N_w):
        # input M size = M_h * M_w
        # input N size = M_w * N_w
        # output P size = M_h * N_w
        block = self.block
        blocksize = (block, block, 1)
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

    def multi_naive_restrict(self, M, N, M_w, M_h, N_w):
        # input M size = M_h * M_w
        # input N size = M_w * N_w
        # output P size = M_h * N_w
        block = self.block
        blocksize = (block, block, 1)
        P = np.zeros(shape=(M_h, N_w)).astype(np.float32)  # create output array in host
        func = self.mod.get_function("mul_naive_res")
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
        block = self.block
        blocksize = (block, block, 1)
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

    def multi_tile_res(self, M, N, M_w, M_h, N_w):
        block = self.block
        blocksize = (block, block, 1)
        P = np.zeros(shape=(M_h, N_w)).astype(np.float32)  # create output array in host
        func = self.mod.get_function("mul_shared_res")
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



