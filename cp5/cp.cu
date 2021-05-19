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
#include <cmath>
#include <bits/stdc++.h>

using namespace std;


__global__ void mykernel(float* result, const float* data, int nx, int ny, int nn) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= ny || j >= ny || j > i)
    return;
    float newValue = 0;
    for(int x = 0; x < nx; x++){
        newValue += data[i + x*nx] * data[x + j*nx];
    }
    result[i + j*ny] = newValue;

}
__global__ void myppkernel(float* result, float* data, float* processedData, int nx, int ny, int nn) {
    int ja = threadIdx.x;
    int i = blockIdx.y;
    
    __shared__ float tempArray[64];

    tempArray[ja] = 0;
    for(int x = 0; x < nn; x+=64){
        int j = ja + x;
        tempArray[ja] += j < nx && i < ny? data[j + i*nx] : 0;
    }
    __syncthreads();

    float averageCalculated = 0;
    for (int x = 0; x < 64; x++) {
        averageCalculated += tempArray[x];
    }
    averageCalculated = averageCalculated / nx;

    __syncthreads();

    tempArray[ja] = 0;
    for(int x = 0; x < nn; x+=64){
        int j = ja + x;
        float newValue = j < nx && i < ny ? (float)data[j + i*nx] - averageCalculated : 0;
        processedData[j + i*nn] = newValue;
        float square = newValue * newValue;
        tempArray[ja] += square;    
    }

    __syncthreads();

    float squareSumCalculated = 0;
    for (int x = 0; x < 64; x++) {
        squareSumCalculated += tempArray[x];
    }
    
    __syncthreads();

    if(squareSumCalculated == 0) {
        squareSumCalculated = 1;
    }
    float* t = processedData + nn * nn;

    for(int x = 0; x < nn; x+=64){
        int j = ja + x;
        float square = (float)sqrt(squareSumCalculated);
        float newValue = (float)processedData[j + i*nn] / square;
        processedData[j + i*nn] = newValue;
        t[i + j*nn] = newValue;
    }

    __syncthreads();
    
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
    int nMax = max(ny, nx);
    int nn = roundup(nMax, 64);

    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, ny * nx * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    float* dProcessedGPU = NULL;
    CHECK(cudaMalloc((void**)&dProcessedGPU, 2 * nn * nn * sizeof(float)));
	
    CHECK(cudaMemset(rGPU, 0, ny * ny * sizeof(float)));
    CHECK(cudaMemset(dProcessedGPU, 0, 2 * nn * nn * sizeof(float)));

    CHECK(cudaMemcpy(dGPU, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));  

    {
        dim3 dimBlock(64, 1);
        dim3 dimGrid(1, nn);
        myppkernel<<<dimGrid, dimBlock>>>(rGPU, dGPU, dProcessedGPU, nx, ny, nn);
        CHECK(cudaGetLastError());
    }
    // Run kernel
    {
    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));
    //dim3 dimGrid( nn / 64, nn / 64);
    mykernel<<<dimGrid, dimBlock>>>(rGPU, dProcessedGPU, nx, ny, nn);
    CHECK(cudaGetLastError());
    }
    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(dProcessedGPU));
    CHECK(cudaFree(rGPU));
    //delete[] newData;
}
