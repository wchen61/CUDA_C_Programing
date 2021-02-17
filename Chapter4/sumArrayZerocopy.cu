#include "../common.h"


/*void initialData(float *ip, int size) {
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
            return false;
        }
        //printf("%d : %f %f\n", idx, A[idx], B[idx]);
    }
    return true;
}*/

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArrays(float *A, float *B, float*C, int nElem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nElem)
        C[idx] = A[idx] + B[idx];
}

__global__ void sumArraysZeroCopy(float *A, float *B, float*C, int nElem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nElem)
        C[idx] = A[idx] + B[idx];
}

int main(int argc, char **argv) {
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    if (!deviceProp.canMapHostMemory) {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
    printf("Using Device %d : %s\n", dev, deviceProp.name);

    int ipower = 10;
    if (argc > 1) ipower = atoi(argv[1]);
    int nElem = 1 << ipower;
    size_t nBytes = nElem * sizeof(float);
    if (ipower < 18) {
        printf("Vector size %d power %d nbytes %3.0f KB\n", nElem, ipower, (float)nBytes / (1024.0f));
    } else {
        printf("Vector size %d power %d nbytes %3.0f MB\n", nElem, ipower, (float)nBytes / (1024.0f * 1024.0f));
    }

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    int iLen = 512;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);

    sumArrays<<<grid, block>>>(d_A, d_B, d_C, nElem);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    checkResult(hostRef, gpuRef, nElem);

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    unsigned int flags = cudaHostAllocMapped;
    cudaHostAlloc((void**)&h_A, nBytes, flags);
    cudaHostAlloc((void**)&h_B, nBytes, flags);
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    //cudaHostGetDevicePointer((void**)&d_A, (void*)h_A, 0);
    //cudaHostGetDevicePointer((void**)&d_B, (void*)h_B, 0);

    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    //sumArraysZeroCopy<<<grid, block>>>(d_A, d_B, d_C, nElem);
    sumArraysZeroCopy<<<grid, block>>>(h_A, h_B, d_C, nElem);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    checkResult(hostRef, gpuRef, nElem);
    cudaFree(d_C);
    cudaFreeHost(d_A);
    cudaFreeHost(d_B);

    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}