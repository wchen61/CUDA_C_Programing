#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define CPU     0
#define GPU1D   0
#define GPU2D   1
#define GPUMix  0 

#define CHECK(call)                                                              \
{                                                                                \
    const cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                                  \
        printf("Error: %s : %d,", __FILE__, __LINE__);                           \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));       \
        exit(1);                                                                 \
    }                                                                            \
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned int) time(&t));
    for (int i=0; i<size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

bool checkResult(float *A, float *B, int size) {
    double epsilon = 1.0E-8;
    for (int idx = 0; idx < size; idx++) {
        if (abs(A[idx] - B[idx]) > epsilon) {
            printf("CheckResult Failed %d : %f %f\n", idx, A[idx], B[idx]);
            return false;
        }
        //printf("%d : %f %f\n", idx, A[idx], B[idx]);
    }
    printf("CheckResult Success\n");
    return true;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny) {
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx; ib += nx; ic += nx;
    }
}

__global__ void sumMatrixOnGPU1D(float *MatA, float *MatB, float *MatC, const int nx, const int ny) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix < nx) {
        for (int iy = 0; iy < ny; iy++) {
            int idx = iy * nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }
    }
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, const int nx, const int ny) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }

    /*if (blockIdx.x == 0 && blockIdx.y == 0) {
        printf("(%d %d) %f + %f -> %f\n", ix, iy, MatA[idx], MatB[idx], MatC[idx]);
    }*/
}

__global__ void sumMatrixOnGPUMix(float *MatA, float *MatB, float *MatC, const int nx, const int ny) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 1 << 14;
    int ny = 1 << 13;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d, ny %d\n", nx, ny);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    initialData(h_A, nxy);
    initialData(h_B, nxy);

    double iStart, iElaps;

#if CPU
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    iStart = cpuSecond();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = cpuSecond() - iStart;
    printf("sumMatrixOnHost %d x %d elapsed %f sec\n", nx, ny, iElaps);
#endif

    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void**)&d_MatA, nBytes));
    CHECK(cudaMalloc((void**)&d_MatB, nBytes));
    CHECK(cudaMalloc((void**)&d_MatC, nBytes));
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

#if GPU2D
    int dimx = 16;
    int dimy = 16;
    if (argc > 2) {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    iStart = cpuSecond();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sumMatrixOnGPU2D <<<(%d, %d), (%d, %d)>>> elapsed %f ms\n",
             grid.x, grid.y, block.x, block.y, iElaps);
#elif GPU1D
    dim3 block(256, 1);
    dim3 grid((nx + block.x - 1) / block.x, 1);

    iStart = cpuSecond();
    sumMatrixOnGPU1D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sumMatrixOnGPU1D <<<(%d, %d), (%d, %d)>>> elapsed %f sec\n",
             grid.x, grid.y, block.x, block.y, iElaps);
#elif GPUMix
    dim3 block(256);
    dim3 grid((nx + block.x - 1) / block.x, ny);

    iStart = cpuSecond();
    sumMatrixOnGPUMix<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sumMatrixOnGPUMix <<<(%d, %d), (%d, %d)>>> elapsed %f sec\n",
             grid.x, grid.y, block.x, block.y, iElaps);
#endif

    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

//    checkResult(gpuRef, hostRef, nxy);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    cudaDeviceReset();
    return 0;
}