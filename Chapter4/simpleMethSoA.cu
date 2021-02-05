#include "../common.h"

#define N (1<<20)

struct innerArray {
    float x[N];
    float y[N];
};

void initialInnerArray(innerArray *ip, int size) {
    for (int i = 0; i < size; i++) {
        ip->x[i] = (float)(rand()&0xFF) / 100.0f;
        ip->y[i] = (float)(rand()&0xFF) / 100.0f;
    }
}

__global__ void warmup(innerArray *data, innerArray *result, const int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float tmpX = data->x[i];
        float tmpY = data->y[i];
        tmpX += 10.f;
        tmpY += 20.f;
        result->x[i] = tmpX;
        result->y[i] = tmpY;
    }
}

__global__ void testInnerArray(innerArray *data, innerArray *result, const int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float tmpX = data->x[i];
        float tmpY = data->y[i];
        tmpX += 10.f;
        tmpY += 20.f;
        result->x[i] = tmpX;
        result->y[i] = tmpY;
        /*if (i < 10)
            printf("%d: %f, %f\n", i, data->x[i], data->y[i]);*/
    }
}

void testInnerArrayHost(innerArray *data, innerArray *result, const int n) {
    for (int i=0; i<n; i++) {
        float tmpX = data->x[i];
        float tmpY = data->y[i];
        tmpX += 10.f;
        tmpY += 20.f;
        result->x[i] = tmpX;
        result->y[i] = tmpY;
    }
}

bool checkInnerArray(innerArray *A, innerArray *B, const int n) {
    double delta = 1.0E-6;
    for (int i=0; i<n; i++) {
        if ((abs(A->x[i] - B->x[i]) > delta) || (abs(A->y[i] - B->y[i]) > delta)) {
            printf("%d Not Match -> X %f : %f, Y %f : %f\n", i, A->x[i], B->x[i], A->y[i], B->y[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s test struct of array at device %d: %s\n", argv[0], dev, deviceProp.name);
    cudaSetDevice(dev);

    int nElem = N;
    size_t nBytes = sizeof(innerArray);
    innerArray *h_A = (innerArray*)malloc(nBytes);
    innerArray *hostRef = (innerArray*)malloc(nBytes);
    innerArray *gpuRef = (innerArray*)malloc(nBytes);

    initialInnerArray(h_A, nElem);
    testInnerArrayHost(h_A, hostRef, nElem);

    innerArray *d_A, *d_C;
    cudaMalloc((innerArray**)&d_A, nBytes);
    cudaMalloc((innerArray**)&d_C, nBytes);

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
    checkInnerArray(hostRef, gpuRef, nElem);

    iStart = cpuSecond();
    testInnerArray<<<grid, block>>>(d_A, d_C, nElem);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("testInnerArray <<< %d, %d >>> elapsed %f sec\n", grid.x, block.x, iElaps);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkInnerArray(hostRef, gpuRef, nElem);

    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}