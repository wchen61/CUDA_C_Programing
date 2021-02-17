#include "../common.h"

void transposeHost(float *out, float *in, const int nx, const int ny) {
    for (int iy = 0; iy < ny; ++iy) {
        for ( int ix = 0; ix < nx; ++ix) {
            out[ix*ny+iy] = in[iy*nx+ix];
        }
    }
}

__global__ void warmup(float *out, float *in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}


__global__ void copyRow(float *out, float *in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

__global__ void copyCol(float *out, float *in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[ix * ny + iy];
    }
}

__global__ void transposeNaiveRow(float *out, float *in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

__global__ void transposeNaiveCol(float *out, float *in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

__global__ void transposeUnroll4Row(float *out, float *in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int ti = iy * nx + ix;
    unsigned int to = ix * ny + iy;

    if (ix + 3 * blockDim.x < nx && iy < ny) {
        out[to] = in[ti];
        out[to + ny * blockDim.x] = in[ti + blockDim.x];
        out[to + 2 * ny * blockDim.x] = in[ti + 2 * blockDim.x];
        out[to + 3 * ny * blockDim.x] = in[ti + 3 * blockDim.x];
    }
}

__global__ void transposeUnroll4Col(float *out, float *in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int to = iy * nx + ix;
    unsigned int ti = ix * ny + iy;

    if (ix + 3 * blockDim.x < nx && iy < ny) {
        out[to] = in[ti];
        out[to + blockDim.x] = in[ti + ny * blockDim.x];
        out[to + 2 * blockDim.x] = in[ti + 2 * ny * blockDim.x];
        out[to + 3 * blockDim.x] = in[ti + 3 * ny * blockDim.x];
    }
}

__global__ void transposeDiagonalRow(float *out, float *in, const int nx, const int ny) {
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

__global__ void transposeDiagonalCol(float *out, float *in, const int nx, const int ny) {
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting transpose at device %d:%s ", argv[0], dev, deviceProp.name);
    cudaSetDevice(dev);

    int nx = 1 << 11;
    int ny = 1 << 11;

    int iKernel = 0;
    int blockx = 16;
    int blocky = 16;

    if (argc > 1) iKernel = atoi(argv[1]);
    if (argc > 2) blockx = atoi(argv[2]);
    if (argc > 3) blocky = atoi(argv[3]);
    if (argc > 4) nx = atoi(argv[4]);
    if (argc > 5) ny = atoi(argv[5]);

    printf(" with matrix %d x %d,  with kernel %d\n", nx, ny, iKernel);

    size_t nBytes = nx * ny * sizeof(float);

    dim3 block(blockx, blocky);
    dim3 grid((nx + blockx - 1) / blockx, (ny + blocky - 1) / blocky);

    float *h_A = (float*)malloc(nBytes);
    float *hostRef = (float*)malloc(nBytes);
    float *gpuRef = (float*)malloc(nBytes);

    initialData(h_A, nx * ny);
    transposeHost(hostRef, h_A, nx, ny);
    float *d_A, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    
    double iStart = cpuSecond();
    warmup<<<grid, block>>>(d_C, d_A, nx, ny);
    cudaDeviceSynchronize();
    double iElaps = cpuSecond() - iStart;
    printf("warmup elapsed %f sec \n", iElaps);

    void(*kernel)(float*, float*, int, int);
    char* kernelName;

    switch(iKernel) {
        case 0:
            kernel = &copyRow;
            kernelName = "CopyRow    ";
            break;
        case 1:
            kernel = &copyCol;
            kernelName = "CopyCol    ";
            break;
        case 2:
            kernel = &transposeNaiveRow;
            kernelName = "Transpose Naive Row    ";
            break;
        case 3:
            kernel = &transposeNaiveCol;
            kernelName = "Transpose Naive Col    ";
            break;
        case 4:
            kernel = &transposeUnroll4Row;
            kernelName = "Transpose Unroll 4 Row    ";
            grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
            break;
        case 5:
            kernel = &transposeUnroll4Col;
            kernelName = "Transpose Unroll 4 Col    ";
            grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
            break;
        case 6:
            kernel = &transposeDiagonalRow;
            kernelName = "Transpose Diagonal Row    ";
            break;
        case 7:
            kernel = &transposeDiagonalCol;
            kernelName = "Transpose Diagonal Col    ";
            break;
    }

    iStart = cpuSecond();
    kernel<<<grid, block>>>(d_C, d_A, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;

    float iBnd = 2 * nx * ny * sizeof(float) / 1e9 / iElaps;
    printf("%s elapsed %f sec <<<grid (%d, %d), block (%d, %d)>>>"
            " effective bandwidth %f GB\n", kernelName, iElaps, grid.x, grid.y,
            block.x, block.y, iBnd);
    
    if (iKernel > 1) {
        cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
        checkResult(hostRef, gpuRef, nx*ny);
    }

    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}