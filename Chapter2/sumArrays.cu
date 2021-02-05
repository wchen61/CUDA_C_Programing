#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

#define CHECK(call)                                                              \
{                                                                                \
    const cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                                  \
        printf("Error: %s : %d,", __FILE__, __LINE__);                           \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));       \
        exit(1);                                                                 \
    }                                                                            \
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

bool checkResult(float *A, float *B, int size) {
    double epsilon = 1.0E-8;
    for (int idx = 0; idx < size; idx++) {
        if (abs(A[idx] - B[idx]) > epsilon) {
            return false;
        }
        //printf("%d : %f %f\n", idx, A[idx], B[idx]);
    }
    return true;
}

__global__ void sumArraysOnDevice(float *A, float *B, float*C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    C[idx] = A[idx] + B[idx];
}

void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned int) time(&t));
    for (int i=0; i<size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}



int main(int argc, char **argv) {
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    float *h_C1;

    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    h_C = (float*)malloc(nBytes);
    h_C1 = (float*)malloc(nBytes);
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    sumArraysOnHost(h_A, h_B, h_C, nElem);

    sumArraysOnDevice<<<1, nElem>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C1, d_C, nBytes, cudaMemcpyDeviceToHost);

    if (!checkResult(h_C, h_C1, nElem)) {
        printf("Result is not identity!\n");
    } else {
        printf("Result is identity!\n");
    }

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}