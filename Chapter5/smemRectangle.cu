#include "../common.h"
#include <cuda_runtime.h>

#define BDIMX 32 
#define BDIMY 16 
#define IPAD 1

void printData(char* msg, int *in, const int size) {
    printf("%s: ", msg);
    for (int i=0; i<size; i++) {
        printf("%5d", in[i]);
        fflush(stdout);
    }
    printf("\n");
    return;
}

__global__ void setRowReadCol(int *out) {
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();
    out[idx] = tile[icol][irow];
}

int main(int argc, char** argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s at ", argv[0]);
    printf("device %d: %s", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    cudaSharedMemConfig pConfig;
    CHECK(cudaDeviceGetSharedMemConfig(&pConfig));
    printf(" with Bank Mode : %s", pConfig == 1 ? "4-Bytes" : "8-Bytes");

    int nx = BDIMX;
    int ny = BDIMY;
    bool iprintf = 0;

    if (argc > 1) iprintf = atoi(argv[1]);
    size_t nBytes = nx * ny * sizeof(int);

    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1);

    printf(" <<<grid (%d, %d) block (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

    int* d_out;
    int* gpuRef;
    CHECK(cudaMalloc((int**)&d_out, nBytes));

    gpuRef = (int*)malloc(nBytes);

    CHECK(cudaMemset(d_out, 0, nBytes));
    setRowReadCol<<<grid, block>>>(d_out);
    CHECK(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));
    if (iprintf) printData("setRowReadCol\t\t\t", gpuRef, nx * ny);
}

