#include "../common.h"

#define DIMX 128
#define DIMY 128 

#define BDIMX 16
#define BDIMY 16
#define IPAD 2

__global__ void naiveGmem(float* out, float* in, const int nx, const int ny) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix*ny+iy] = in[iy*nx+ix];
    }
}

__global__ void copyGmem(float* out, float *in, const int nx, const int ny) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}


__global__ void transposeSmem(float *out, float *in, int nx, int ny) {
    __shared__ float tile[BDIMY][BDIMX];

    unsigned int ix, iy, ti, to;
    ix = blockIdx.x * blockDim.x + threadIdx.x;
    iy = blockIdx.y * blockDim.y + threadIdx.y;

    ti = iy * nx + ix;
    unsigned int bidx, irow, icol;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    ix = blockIdx.y * blockDim.y + icol;
    iy = blockIdx.x * blockDim.x + irow;
    to = iy * ny +  ix;
    if (ix < nx && iy < ny) {
        tile[threadIdx.y][threadIdx.x] = in[ti];
        __syncthreads();
        out[to] = tile[icol][irow];
    }
}

__global__ void transposeSmemUnrollPad(float *out, float *in, const int nx, const int ny) {
    __shared__ float tile[BDIMY * (BDIMX * 2 + IPAD)];
    unsigned int ix = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int ti = iy * nx + ix;
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    unsigned int ix2 = blockIdx.y * blockDim.y + icol;
    unsigned int iy2 = 2 * blockIdx.x * blockDim.x + irow;

    unsigned int to = iy2 * ny + ix2;

    if (ix + blockDim.x < nx && iy < ny) {
        unsigned int row_idx = threadIdx.y * (blockDim.x * 2 + IPAD) + threadIdx.x;
        tile[row_idx] = in[ti];
        tile[row_idx + BDIMX] = in[ti+BDIMX];

        __syncthreads();
        unsigned int col_idx = icol * (blockDim.x * 2 + IPAD) + irow;
        out[to] = tile[col_idx];
        out[to + ny * BDIMX] = tile[col_idx + BDIMX];
    }
}


int main(int argc, char** argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting transposeSmem at ", argv[0]);
    printf("device %d: %s \n", dev, deviceProp.name);
    cudaSetDevice(dev);

    bool iprintf = 0;
    if (argc > 1) iprintf = atoi(argv[1]);

    int nx = BDIMX;
    int ny = BDIMY;
    dim3 block(nx, ny);
    dim3 grid(1, 1);

    size_t bytes = nx * ny * sizeof(float);
    int size = nx * ny;
    float *h_idata = (float*)malloc(bytes);
    float *h_odata = (float*)malloc(bytes);
    int *tmp = (int*)malloc(bytes);

    srand((unsigned)time(NULL));
    for (int i = 0; i < size; i++) {
        float x = rand() % 10;
        x = x / 10.0f;
        h_idata[i] = x;
    }
    if (iprintf) printData("Origin Data: ", h_idata, nx, ny);

    double iStart, iElaps;
    float *d_idata = NULL;
    float *d_odata = NULL;
    CHECK(cudaMalloc(&d_idata, bytes));
    CHECK(cudaMalloc(&d_odata, bytes));

    /*iStart = cpuSecond();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = cpuSecond() - iStart;
    printf("cpu reduce elapsed: %f s, cpu_sum: %d\n", iElaps, cpu_sum);*/

    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = cpuSecond();
    naiveGmem<<<grid, block>>>(d_odata, d_idata, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, bytes, cudaMemcpyDeviceToHost));
    printf("navieGmem elapsed: %f s\n", iElaps);
    if (iprintf) printData("naiveGmem: ", h_odata, nx, ny);

    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = cpuSecond();
    copyGmem<<<grid, block>>>(d_odata, d_idata, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, bytes, cudaMemcpyDeviceToHost));
    printf("copyGmem elapsed: %f s\n", iElaps);
    if (iprintf) printData("copyGmem: ", h_odata, nx, ny);

    block = dim3(BDIMX, BDIMY);
    grid = dim3((DIMX + block.x - 1) / block.x, (DIMY + block.y - 1) / block.y);
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = cpuSecond();
    transposeSmem<<<grid, block>>>(d_odata, d_idata, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, bytes, cudaMemcpyDeviceToHost));
    printf("transposeSmem elapsed: %f s\n", iElaps);
    if (iprintf) printData("transposeGmem: ", h_odata, nx, ny);
    
    block = dim3(BDIMX, BDIMY);
    grid = dim3((DIMX + block.x - 1) / block.x, (DIMY + block.y - 1) / block.y);
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = cpuSecond();
    transposeSmemUnrollPad<<<grid, block>>>(d_odata, d_idata, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, bytes, cudaMemcpyDeviceToHost));
    printf("transposeSmemUnrollPad elapsed: %f s\n", iElaps);
    if (iprintf) printData("transposeGmemUnrollPad: ", h_odata, nx, ny);
}