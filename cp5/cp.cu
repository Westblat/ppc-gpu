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


__global__ void mykernel(float* result, const float* data, int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= ny || j >= ny || j > i)
    return;
    float size = nx;
    //printf("%d i %d j ", i, j);
    double newValue = 0;
    for(int x = 0; x < nx; x++){
        //printf("%d i %d j ", data[x + i*nx], data[x + j*nx]);
        newValue += data[x + i*nx] * data[x + j*nx];
    }
    result[i + j*ny] = newValue;
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
    vector<float> averageList(ny, 0);
    float *newData = new float[nx*ny];
    vector<float> squareSums(ny, 0);

    for(int i = 0; i < ny; i++){
        float average = 0;
        for(int j = 0; j < nx; j++){
            average += (float)data[j + i*nx];
        }
        average = average / (float)nx;
        averageList[i] = average;
    }

    for(int i = 0; i < ny; i++){
        float rowSquareSum = 0;
        for(int j = 0; j < nx; j++){
            float newValue = (float)data[j + i*nx] - averageList[i];
            newData[j + i*nx] = newValue;
            float square = newValue * newValue;
            rowSquareSum += square;
        }
        squareSums[i] = rowSquareSum;
    }

    for(int i = 0; i < ny; i++){
        for(int j = 0; j < nx; j++){
            float square = (float)sqrt(squareSums[i]);
            float newValue = newData[j + i*nx] / square;
            newData[j + i*nx] = (float)newValue;
        }
    }
	
    CHECK(cudaMemset(rGPU, 0, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, newData, ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));
    mykernel<<<dimGrid, dimBlock>>>(rGPU, dGPU, nx, ny);
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
    delete[] newData;
}
