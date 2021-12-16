#include <stdio.h>

//Some pre-defined arguments
#define TILE_WIDTH 32

// functions for global id calculating
/**
 * @brief calculate the global id of coordinate (n1,n2) in linearized 2-dimensional 
 * matrix based on row-major layout
 * @param n1 coordinate in direction n1
 * @param n2 coordinate in direction n2
 * @param N2 length of the matrix in direction n2
 * @return global id of (n1,n2)
**/
__device__ int global_id_2d(int n1, int n2, int N2){
    return n2 + N2*n1;
}

// functions for global id calculating
/**
 * @brief calculate the global id of coordinate (n1,n2,n3) in linearized 3-dimensional 
 * matrix based on row-major layout
 * @param n1 coordinate in direction n1
 * @param n2 coordinate in direction n2
 * @param n3 coordinate in direction n3
 * @param N2 length of the matrix in direction n2
 * @param N3 length of the matrix in direction n3
 * @return global id of (n1,n2,n3)
**/
__device__ int global_id_3d(int n1, int n2, int n3, int N2, int N3){
    return n3 + N3*global_id_2d(n1,n2,N2);
    //return n3 + N3*(n2 + N2*n1);
}

/**
 * @brief calculate the global id of coordinate (n1,n2,n3,n4) in linearized 4-dimensional 
 * matrix based on row-major layout
 * @param n1 coordinate in direction n1
 * @param n2 coordinate in direction n2
 * @param n3 coordinate in direction n3
 * @param n4 coordinate in direction n4
 * @param N2 length of the matrix in direction n2
 * @param N3 length of the matrix in direction n3
 * @param N4 length of the matrix in direction n4
 * @return global id of (n1,n2,n3,n4)
**/
__device__ int global_id_4d(int n1, int n2, int n3, int n4, int N2, int N3, int N4){
    return n4 + N4*global_id_3d(n1,n2,n3,N2,N3);
    //return n4 + N4*(n3 + N3*(n2 + N2*n1));
}


/**
 * @brief Naive parallel convolution layer without using shared or constant memory. 
 * mode = valid, stride = 1, mask_width = K.
 * @param X input matrix with size [N, C, H, W]
 * @param Masks masks with size [M, C, K, K]
 * @param Y output matrix with size [N, M, H-K+1, W-K+1]
 * @param N number of samples 
 * @param C number of channels of input matrix
 * @param M number of channels of output matrix
 * @param H height of input matrix
 * @param W width of input matrix
 * @param K width of masks 
 * @param W_grid the number of tiled matrix in width direction
 * @return Convolution result filled in Y
**/
__global__ void convLayer_forward_naive(
    float *X, 
    float *Masks, 
    float *Y, 
    const int N, 
    const int C, 
    const int M, 
    const int H, 
    const int W, 
    const int K,
    const int W_grid){

    // output shape of Y
    const int h_y = H-K+1;
    const int w_y = W-K+1; 

    // initialize some parameters
    int c, p, q;
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    // const int h = blockIdx.z / W_grid + threadIdx.y;
    // const int w = blockIdx.z % W_grid + threadIdx.x;
    const int h = (blockIdx.z / W_grid)*blockDim.y + threadIdx.y;
    const int w = (blockIdx.z % W_grid)*blockDim.x + threadIdx.x;

    float acc = 0;
    // for each input channel
    for(c=0; c<C; c++)
        // convolution
        for(p=0; p<K; p++)          // y-direction
            for(q=0; q<K; q++){      // x-direction
                int gid_x = global_id_4d(n, c, h+p, w+q, C, H, W);
                int gid_m = global_id_4d(m, c, p, q, C, K, K);
                acc += X[gid_x] * Masks[gid_m];
            }
    int gid_y = global_id_4d(n, m, h, w, M, h_y, w_y);
    Y[gid_y] = acc;
}


/**
 * @brief Naive parallel convolution layer without using shared or constant memory. 
 * mode = valid, stride = 1, mask_width = K.
 * @param X input matrix with size [N, H, W, C]
 * @param Masks masks with size [K, K, C, M]
 * @param Y output matrix with size [N, H-K+1, W-K+1, M]
 * @param N number of samples 
 * @param C number of channels of input matrix
 * @param M number of channels of output matrix
 * @param H height of input matrix
 * @param W width of input matrix
 * @param K width of masks 
 * @param W_grid the number of tiled matrix in width direction
 * @return Convolution result filled in Y
**/
__global__ void convLayer_forward_naive_channel(
    float *X, 
    float *Masks, 
    float *Y, 
    int N, 
    int C, 
    int M, 
    int H, 
    int W, 
    int K,
    const int W_grid){

    // output shape of Y
    const int h_y = H-K+1;
    const int w_y = W-K+1; 

    // initialize some parameters
    int c, p, q;
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    const int h = blockIdx.z / W_grid + threadIdx.y;
    const int w = blockIdx.z % W_grid + threadIdx.x;

    float acc = 0;
    // for each input channel
    for(c=0; c<C; c++)
        // convolution
        for(p=0; p<K; p++)          // y-direction
            for(q=0; q<K; q++){      // x-direction
                int gid_x = global_id_4d(n, h+p, w+q, c, H, W, C);
                int gid_m = global_id_4d(p, q, c, m, K, C, M);
                acc += X[gid_x] * Masks[gid_m];
            }
    int gid_y = global_id_4d(n, h, w, m, h_y, w_y, M);
    Y[gid_y] = acc;
}


/**
 * @brief parallel convolution layer using shared memory. 
 * mode = valid, stride = 1, mask_width = K.
 * @param X input matrix with size [N, C, H, W]
 * @param Masks masks with size [M, C, K, K]
 * @param Y output matrix with size [N, M, H-K+1, W-K+1]
 * @param N number of samples 
 * @param C number of channels of input matrix
 * @param M number of channels of output matrix
 * @param H height of input matrix
 * @param W width of input matrix
 * @param K width of masks 
 * @param W_grid the number of tiled matrix in width direction
 * @return Convolution result filled in Y
**/
__global__ void convLayer_forward_shared(
    float *X, 
    float *Masks, 
    float *Y, 
    const int C, 
    const int M, 
    const int H, 
    const int W, 
    const int K,
    const int W_grid){

    // the size to be tiled for X matrix
    const int X_tile_width = TILE_WIDTH + K - 1;
    // allocate shared memory, shared memory size defined when invoking the kernel
    // extern __shared__ float shmem[];
    __shared__ float shmem[1321];
    // first part of shared memory is tile of X, 
    // X_tile has size X_tile_width*X_tile_width
    float *X_shared = &shmem[0];
    // second part of shared memory is part of the mask
    // has size K*K
    // float *Mask_shared = &shmem[X_tile_width*X_tile_width];
    float *Mask_shared = &shmem[1296];
    // Mask_shared[24] = 1;

    // output shape of Y
    const int h_y = H-K+1;
    const int w_y = W-K+1; 

    // initialize some parameters
    // int n, m, h0, w0, h_base, w_base, h, w;
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    const int h0 = threadIdx.x;
    const int w0 = threadIdx.y;
    const int h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    const int w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    const int h = h_base + h0;
    const int w = w_base + w0;


    float acc = 0;

    int c, i, j, p, q;
    // for each input channel
    // update the shared memory in each iteration
    for(c=0; c<C; c++){

        // copy mask[m,c,:,:] to the shared memory
        // here h0 = threadIdx.x, w0 = threadIdx.y
        if((h0<K) && (w0<K))
            Mask_shared[global_id_2d(h0,w0,K)] = Masks[global_id_4d(m,c,h0,w0,C,K,K)];
        __syncthreads();

        // copy tiled X to the shared memory
        for(i=h; i<(h_base + X_tile_width); i+=TILE_WIDTH)
            for(j=w; j<(w_base + X_tile_width); j+=TILE_WIDTH)
                if(i<H && j<W)
                    X_shared[global_id_2d(i-h_base,j-w_base,X_tile_width)] = X[global_id_4d(n,c,i,j,C,H,W)];
        __syncthreads();

        // convolution
        for(p=0; p<K; p++)
            for(q=0; q<K; q++)
                acc += X_shared[h+p,w+q] * Mask_shared[p,q];
        __syncthreads();
    }
    Y[global_id_4d(n, m, h, w, M, h_y, w_y)] = acc;
}