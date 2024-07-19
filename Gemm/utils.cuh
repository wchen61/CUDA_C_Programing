#pragma once
#include <iostream>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define checkCudaErrors(call)                                                           \
    {                                                                                   \
        cudaError_t err = call;                                                         \
        if (err != cudaSuccess) {                                                       \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__        \
                    << ": " << cudaGetErrorString(err) << std::endl;                    \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    }

template <typename T>
void printData(char* msg, T *in, const int x, const int y) {
    std::cout << msg << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    for (int i=0; i<y; i++) {
        for (int j=0; j<x; j++) {
            std::cout << std::setw(8) << in[i*x+j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    return;
}

void cpuSgemm(float *a, float *b, float *c, const int M, const int N, const int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

float testError(
    void (*gpuSgemm) (float*, float*, float*, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float*)malloc(size_a);
    h_b = (float*)malloc(size_b);
    h_c = (float*)malloc(size_c);
    checkCudaErrors(cudaMalloc(&d_a, size_a));
    checkCudaErrors(cudaMalloc(&d_b, size_b));
    checkCudaErrors(cudaMalloc(&d_c, size_c));
    h_d_c = (float*)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        h_b[i] = (float)rand() / RAND_MAX;
    }
    checkCudaErrors(cudaMemset(d_c, 0, size_c));
    cpuSgemm(h_a, h_b, h_c, M, N, K);
    /*printData("A", h_a, K, M);
    printData("B", h_b, N, K);
    printData("C", h_c, N, M);*/
 
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    //printData("d_C", h_d_c, N, M);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) {
            max_error = -NAN;
        } else {
            max_error = max(max_error, this_error);
        }
    }


    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);
    return max_error;
}

float testPerformance(
    void (*gpuSgemm)(float*, float*, float*, int, int, int),
    dim3 gridDim, dim3 blockDim, int M, int N, int K, int repeat) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    for (int i = 0; i < repeat; i++) {
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
        //checkCudaErrors(cudaDeviceSynchronize());
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return sec;
}
