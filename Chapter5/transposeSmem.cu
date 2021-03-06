#include "../common.h"

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


#define DIMX 32 
#define DIMY 32

int main(int argc, char** argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting transposeSmem at ", argv[0]);
    printf("device %d: %s \n", dev, deviceProp.name);
    cudaSetDevice(dev);

    bool iprintf = 0;
    if (argc > 1) iprintf = atoi(argv[1]);

    int nx = DIMX;
    int ny = DIMY;
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
}