#include "../common.h"

struct innerStruct {
    float x;
    float y;
};

void initialInnerStruct(innerStruct *ip, int size) {
    for (int i = 0; i < size; i++) {
        ip[i].x = (float)(rand()&0xFF) / 100.0f;
        ip[i].y = (float)(rand()&0xFF) / 100.0f;
    }
}

__global__ void warmup(innerStruct *data, innerStruct *result, const int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        innerStruct temp = data[i];
        temp.x += 10.0f;
        temp.y += 20.0f;
        result[i] = temp;
    }
}

__global__ void testInnerStruct(innerStruct *data, innerStruct *result, const int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        innerStruct temp = data[i];
        temp.x += 10.0f;
        temp.y += 20.0f;
        result[i] = temp;
    }
}

void testInnerStructHost(innerStruct *data, innerStruct *result, const int n) {
    for (int i=0; i<n; i++) {
        innerStruct temp = data[i];
        temp.x += 10.0f;
        temp.y += 20.0f;
        result[i] = temp;
    }
}

bool checkInnerStruct(innerStruct *A, innerStruct *B, const int n) {
    double delta = 1.0E-6;
    for (int i=0; i<n; i++) {
        if ((abs(A[i].x - B[i].x) > delta) || (abs(A[i].y - B[i].y) > delta)) {
            printf("%d Not Match -> X %f : %f, Y %f : %f\n", i, A[i].x, B[i].x, A[i].y, B[i].y);
            return false;
        }
    }
    return true;
}

#define LEN (1<<20)

int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s test struct of array at device %d: %s\n", argv[0], dev, deviceProp.name);
    cudaSetDevice(dev);

    int nElem = LEN;
    size_t nBytes = nElem * sizeof(innerStruct);
    innerStruct *h_A = (innerStruct*)malloc(nBytes);
    innerStruct *hostRef = (innerStruct*)malloc(nBytes);
    innerStruct *gpuRef = (innerStruct*)malloc(nBytes);

    initialInnerStruct(h_A, nElem);
    testInnerStructHost(h_A, hostRef, nElem);

    innerStruct *d_A, *d_C;
    cudaMalloc((innerStruct**)&d_A, nBytes);
    cudaMalloc((innerStruct**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    int blockSize = 128;
    if (argc > 1) blockSize = atoi(argv[1]);
    dim3 block(blockSize, 1);
    dim3 grid((nElem + block.x -1) / block.x);

    double iStart = cpuSecond();
    warmup<<<grid, block>>>(d_A, d_C, nElem);
    cudaDeviceSynchronize();
    double iElaps = cpuSecond() - iStart;
    printf("Warmup <<< %d, %d >>> elapsed %f sec\n", grid.x, block.x, iElaps);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkInnerStruct(hostRef, gpuRef, nElem);

    iStart = cpuSecond();
    testInnerStruct<<<grid, block>>>(d_A, d_C, nElem);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("testInnerStruct <<< %d, %d >>> elapsed %f sec\n", grid.x, block.x, iElaps);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkInnerStruct(hostRef, gpuRef, nElem);

    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}