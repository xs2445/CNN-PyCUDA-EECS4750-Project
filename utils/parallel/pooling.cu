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