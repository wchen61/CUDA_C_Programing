#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#ifndef __COMMON_H__
#define __COMMON_H__

#define CHECK(call)                                                              \
{                                                                                \
    const cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                                  \
        printf("Error: %s : %d,", __FILE__, __LINE__);                           \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));       \
        exit(1);                                                                 \
    }                                                                            \
}

#define CHECK_CUDAERROR()                                                                     \
{                                                                                   \
    const cudaError_t error = cudaGetLastError();                                   \
    if (error != cudaSuccess) {                                                     \
        fprintf(stderr, "Cuda Get Error at %s : %d, error %d : %s\n",               \
            __FILE__, __LINE__, error, cudaGetErrorString(error));                  \
    }                                                                               \
}

#define CHECK_CUBLAS(call)                                                              \
{                                                                                       \
    cublasStatus_t err;                                                                 \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS) {                                      \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__, __LINE__);     \
        exit(1);                                                                        \
    }                                                                                   \
}

#define CHECK_CURAND(call)                                                              \
{                                                                                       \
    curandStatus_t err;                                                                 \
    if ((err = (call)) != CURAND_STATUS_SUCCESS) {                                      \
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__, __LINE__);     \
        exit(1);                                                                        \
    }                                                                                   \
}

#define CHECK_CUFFT(call)                                                              \
{                                                                                      \
    cufftStatus_t err;                                                                 \
    if ((err = (call)) != CUFFT_STATUS_SUCCESS) {                                      \
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__, __LINE__);     \
        exit(1);                                                                       \
    }                                                                                  \
}

#define CHECK_CUSPARSE(call)                                                              \
{                                                                                         \
    cusparseStatus_t err;                                                                 \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS) {                                      \
        fprintf(stderr, "Got CUSPARSE error %d at %s:%d\n", err, __FILE__, __LINE__);     \
        cudaError_t cuda_err = cudaGetLastError();                                        \
        if (cuda_err != cudaSuccess) {                                                    \
            fprintf(stderr, "   CUDA error \" %s \" also detected\n",                     \
                    cudaGetErrorString(cuda_err));                                        \
        }                                                                                 \
        exit(1);                                                                          \
    }                                                                                     \
}

double cpuSecond() {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned int)time(&t));
    for (int i=0; i<size; i++) {
        ip[i] = (float)(rand()&0xFF) / 10.0f;
    }
}

bool checkResult(float *A, float *B, int size) {
    double epsilon = 1.0E-6;
    for (int idx = 0; idx < size; idx++) {
        if (abs(A[idx] - B[idx]) > epsilon) {
            printf("%d NOT MATCH -> %f : %f\n", idx, A[idx], B[idx]);
            return false;
        }
        /*if (idx < 100)
            printf("%d MATCHED -> %f : %f\n", idx, A[idx], B[idx]);*/
    }
    return true;
}

#endif