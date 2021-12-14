#include <stdio.h>


/*********************************************************************/
// 

__global__ 
void convLayer_forward_naive0(float *X, float *Masks, float *Y, int C, int M, int H, int W, int K){
    /*
    Naive parallel convolution without using shared or constant memory,
    the number and shape of threads blocks equals the shape of output matrix

    Properties:
    convolution layer:
    mode = valid
    stride = 1
    mask_width = K

    Args:
    - X: input matrix with size [C, H, W]
    - Masks: masks with size [M, C, K, K]
    - Y: output matrix with size [M, H-K+1, W-K+1]
    - C: number of channels of input matrix
    - M: number of channels of output matrix
    - H: height of input matrix
    - W: width of input matrix
    - K: width of masks 
    */

    // current position of the thread
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    // specify some temporary args
    int m, c, h, w, p, q = 0;
    // the shape of output matrix
    const int H_out = H-K+1;
    const int W_out = W-K+1;

    // for each channel of output matrix Y
    for(m=0; m<M; m++){
        // for each element of output matrix Y
        h = row;
        w = col;

        // do convolution of submatrix and assign to Y[m,h,w]
        Y[m*H_out*W_out + h*W_out + w] = 0;
        // sum the result of each channel of input matrix
        for(c=0; c<C; c++)
            // in place product of X and Masks
            for(p=0; p<K; p++)
                for(q=0; q<K; q++)
                    // result += X[c,h+q,w+q] * Masks[m,c,p,q]
                    Y[m*H_out*W_out + h*W_out + w] += X[c*H*W + h*W + w] *Masks[m*C*K*K + c*K*K + p*K + q];

    }
}

/*********************************************************************/


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








/*********************************************************************/

//Some pre-defined arguments
#define TILE_WIDTH 16

/*********************************************************************/


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
    // for each input chennel
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