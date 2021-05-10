/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

__global__ void mykernel(float* result, const float* data, int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= n || j >= n)
        return;
    float size = nx;
    float sumI = 0;
    float sumJ = 0;
    float sumJI = 0;
    float squareSumI = 0;
    float squareSumJ = 0;    


    for(i=i; i < ny; i++){
        for (j=i; j<ny; j++){
    
            for (int x = 0; x < nx; x++){
                sumI += data[x+i*nx];
                sumJ += data[x+j*nx];
                sumJI += data[x+i*nx] * data[x+j*nx];
                squareSumI += data[x+i*nx] * data[x+i*nx];
                squareSumJ += data[x+j*nx] * data[x+j*nx];
            }
        }
        result[i + j*ny] = (size * sumJI - sumJ * sumI) 
        / sqrt((size * squareSumJ - sumJ * sumJ) * (size * squareSumI - sumI * sumI));
        sumI =0;
        sumJ = 0;
        sumJI = 0;
        squareSumI = 0;
        squareSumJ = 0;

    }
}


static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

#define CHECK(x) check(x, #x)
void correlate(int ny, int nx, const float *data, float *result) {
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, ny * nx * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, ny * nx * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(nx, dimBlock.x), divup(ny, dimBlock.y));
    mykernel<<<dimGrid, dimBlock>>>(rGPU, dGPU, nx, ny);
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, rGPU, nx * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
    
}
