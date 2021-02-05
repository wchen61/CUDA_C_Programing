#include "../common.h"

__global__ void warmup(float *A, float *B, float *C, const int n, int offset) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;
    if (k < n) C[i] = A[k] + B[k];
}

__global__ void readOffset(float *A, float *B, float *C, const int n, int offset) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;
    if (k < n) C[i] = A[k] + B[k];
}

__global__ void readOffsetUnroll4(float *A, float *B, float *C, const int n, int offset) {
    unsigned int i = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    unsigned int k = i + offset;

    if (k + 3 * blockDim.x < n) {
        C[i] = A[k] + B[k];
        C[i + blockDim.x] = A[k + blockDim.x] + B[k + blockDim.x];
        C[i + 2 * blockDim.x] = A[k + 2 * blockDim.x] + B[k + 2 * blockDim.x];
        C[i + 3 * blockDim.x] = A[k + 3 * blockDim.x] + B[k + 3 * blockDim.x];
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int n, int offset) {
    for (int idx = offset, k = 0; idx < n; idx++, k++) {
        C[k] = A[idx] + B[idx]; 
    }
}

int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction at ", argv[0]);
    printf(" device %d: %s ", dev, deviceProp.name);
    cudaSetDevice(dev);

    int nElem = 1 << 20;
    printf(" with array size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    int blockSize = 512;
    int offset = 0;
    if (argc > 1) offset = atoi(argv[1]);
    if (argc > 2) blockSize = atoi(argv[2]);

    dim3 block(blockSize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);

    float *h_A = (float*)malloc(nBytes);
    float *h_B = (float*)malloc(nBytes);
    float *hostRef = (float*)malloc(nBytes);
    float *gpuRef = (float*)malloc(nBytes);

    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);

    sumArraysOnHost(h_A, h_B, hostRef, nElem, offset);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    double iStart = cpuSecond();
    warmup<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    cudaDeviceSynchronize();
    double iElaps = cpuSecond() - iStart;
    printf("Warmup <<<%d, %d>>> offset %d elapsed %f sec\n", grid.x, block.x, offset, iElaps);

    iStart = cpuSecond();
    readOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    iElaps = cpuSecond() - iStart;
    printf("readOffset <<<%d, %d>>> offset %d elapsed %f sec\n", grid.x, block.x, offset, iElaps);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nElem-offset);

    iStart = cpuSecond();
    readOffsetUnroll4<<<grid.x / 4, block>>>(d_A, d_B, d_C, nElem, offset);
    iElaps = cpuSecond() - iStart;
    printf("readOffsetUnroll4 <<<%d, %d>>> offset %d elapsed %f sec\n", grid.x, block.x, offset, iElaps);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nElem-offset);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}