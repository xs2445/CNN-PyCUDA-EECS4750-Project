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



//---------naive restrict-----


__global__ void mul_naive_res(const float* __restrict__ M, const float* __restrict__ N, 
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
//---------naive with restrict ends-------------------------------------------

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

//---------shared_res starts----------------------------------------
#define blocksize 32  // which equals tile size and blockDim.x,y
                        // should be the same in host code
                        
__global__ void mul_shared_res(const float* __restrict__ M, const float* __restrict__ N, 
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
    #pragma unroll
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
        
        #pragma unroll
        for(int j=0;j<blocksize;j++){
            temp += MS[ty][j]*NS[j][tx];
        }
        __syncthreads();
    }
    if(row<Mh && col< Nw){
        P[row*Nw+col] = temp;
    }
}
//----------shared_res ends-----------------------------------------------