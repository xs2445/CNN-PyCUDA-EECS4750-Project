#include <stdio.h>

//Some pre-defined arguments
#define TILE_WIDTH 16


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
    return n3 + N3*(n2 + N2*n1);
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
    return n4 + N4*(n3 + N3*(n2 + N2*n1));
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
 * @return Convolution result filled in Y
**/
__global__ void convLayer_forward_naive(
    float *X, 
    float *Masks, 
    float *Y, 
    int N, 
    int C, 
    int M, 
    int H, 
    int W, 
    int K){

    // output shape of Y
    const int h_y = H-K+1;
    const int w_y = W-K+1; 

    // initialize some parameters
    int c, p, q;
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    const int h = blockIdx.z / TILE_WIDTH + threadIdx.y;
    const int w = blockIdx.z % TILE_WIDTH + threadIdx.x;

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
    int K){

    // output shape of Y
    const int h_y = H-K+1;
    const int w_y = W-K+1; 

    // initialize some parameters
    int c, p, q;
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    const int h = blockIdx.z / TILE_WIDTH + threadIdx.y;
    const int w = blockIdx.z % TILE_WIDTH + threadIdx.x;

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