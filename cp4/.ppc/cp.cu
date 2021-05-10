/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
//#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

using namespace std;


__global__ void mykernel(float* result, const float* data, int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= ny || j >= ny || j > i)
    if(j > i){
        result[i + j*ny] = 0;
    }    
    return;
    float size = nx;
    //printf("%d j %d i", j, i);

    double sumI = 0;
    double sumJ = 0;
    double sumJI = 0;
    double squareSumI = 0;
    double squareSumJ = 0;    
    for (int x = 0; x < nx; x++){
        sumI += (double)data[x+i*nx];
        sumJ += (double)data[x+j*nx];
        sumJI += (double)data[x+i*nx] * (double)data[x+j*nx];
        squareSumI += (double)data[x+i*nx] * (double)data[x+i*nx];
        squareSumJ += (double)data[x+j*nx] * (double)data[x+j*nx];
    }
    double shit = (double)sqrt((size * squareSumJ - sumJ * sumJ) * (size * squareSumI - sumI * sumI));
    double asd = (double)(size * sumJI - sumJ * sumI) / shit;
    result[i + j*ny] = (float)asd;
}


static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(x) check(x, #x)

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

void correlate(int ny, int nx, const float *data, float *result) {
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, ny * nx * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));
    mykernel<<<dimGrid, dimBlock>>>(rGPU, dGPU, nx, ny);
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
    
}
