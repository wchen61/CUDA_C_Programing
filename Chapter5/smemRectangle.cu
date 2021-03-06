#include "../common.h"
#include <cuda_runtime.h>

#define BDIMX 32 
#define BDIMY 16 
#define IPAD 2

void printData(char* msg, int *in, const int size) {
    printf("%s: ", msg);
    for (int i=0; i<size; i++) {
        printf("%5d", in[i]);
        fflush(stdout);
    }
    printf("\n");
    return;
}

__global__ void setRowReadRow (int *out) {
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int *out) {
    __shared__ int tile[BDIMX][BDIMY];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
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

__global__ void setRowReadColDyn(int *out) {
    extern __shared__ int tile[];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    unsigned int col_idx = icol * blockDim.x + irow;

    tile[idx] = idx;

    __syncthreads();

    out[idx] = tile[col_idx];
}

__global__ void setRowReadColPad(int *out) {
    __shared__ int tile[BDIMY][BDIMX + IPAD];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();

    out[idx] = tile[icol][irow];
}

__global__ void setRowReadColDynPad(int *out) {
    extern __shared__ int tile[];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    unsigned int col_idx = icol * (blockDim.x + IPAD) + irow;

    tile[row_idx] = idx;
    __syncthreads();

    out[idx] = tile[col_idx];
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
    setRowReadRow<<<grid, block>>>(d_out);
    CHECK(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));
    if (iprintf) printData("setRowReadRow\t\t\t", gpuRef, nx, ny);

    CHECK(cudaMemset(d_out, 0, nBytes));
    setColReadCol<<<grid, block>>>(d_out);
    CHECK(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));
    if (iprintf) printData("setColReadCol\t\t\t", gpuRef, nx, ny);

    CHECK(cudaMemset(d_out, 0, nBytes));
    setRowReadCol<<<grid, block>>>(d_out);
    CHECK(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));
    if (iprintf) printData("setRowReadCol\t\t\t", gpuRef, ny, nx);

    CHECK(cudaMemset(d_out, 0, nBytes));
    setRowReadColDyn<<<grid, block, BDIMY * BDIMX * sizeof(int)>>>(d_out);
    CHECK(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));
    if (iprintf) printData("setRowReadColDyn\t\t\t", gpuRef, ny, nx);

    CHECK(cudaMemset(d_out, 0, nBytes));
    setRowReadColPad<<<grid, block>>>(d_out);
    CHECK(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));
    if (iprintf) printData("setRowReadColPad\t\t\t", gpuRef, ny, nx);

    CHECK(cudaMemset(d_out, 0, nBytes));
    setRowReadColDynPad<<<grid, block, BDIMY * (BDIMX + IPAD) * sizeof(int)>>>(d_out);
    CHECK(cudaMemcpy(gpuRef, d_out, nBytes, cudaMemcpyDeviceToHost));
    if (iprintf) printData("setRowReadColDynPad\t\t\t", gpuRef, ny, nx);
}

