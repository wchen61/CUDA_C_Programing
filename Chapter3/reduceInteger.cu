#include "../common.h"

__global__ void warmup(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] +=idata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] +=idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
        //printf("block result %d : %d\n", blockIdx.x, g_odata[blockIdx.x]);
    }
}

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n) return;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            idata[index] += idata[index + stride];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceInterLeaved(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n) return;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + 7 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2*blockDim.x];
        int a4 = g_idata[idx + 3*blockDim.x];
        int a5 = g_idata[idx + 4*blockDim.x];
        int a6 = g_idata[idx + 5*blockDim.x];
        int a7 = g_idata[idx + 6*blockDim.x];
        int a8 = g_idata[idx + 7*blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarps8(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + 7 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2*blockDim.x];
        int a4 = g_idata[idx + 3*blockDim.x];
        int a5 = g_idata[idx + 4*blockDim.x];
        int a6 = g_idata[idx + 5*blockDim.x];
        int a7 = g_idata[idx + 6*blockDim.x];
        int a8 = g_idata[idx + 7*blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceCompleteUnrollWarps8(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + 7 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2*blockDim.x];
        int a4 = g_idata[idx + 3*blockDim.x];
        int a5 = g_idata[idx + 4*blockDim.x];
        int a6 = g_idata[idx + 5*blockDim.x];
        int a7 = g_idata[idx + 6*blockDim.x];
        int a8 = g_idata[idx + 7*blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];
    __syncthreads();

    if (blockDim.x >=128 && tid < 64) idata[tid] += idata[tid + 64];
    __syncthreads();

    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + 7 * blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2*blockDim.x];
        int a4 = g_idata[idx + 3*blockDim.x];
        int a5 = g_idata[idx + 4*blockDim.x];
        int a6 = g_idata[idx + 5*blockDim.x];
        int a7 = g_idata[idx + 6*blockDim.x];
        int a8 = g_idata[idx + 7*blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];
    __syncthreads();

    if (blockDim.x >=128 && tid < 64) idata[tid] += idata[tid + 64];
    __syncthreads();

    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

int recursiveReduce(int *data, int const size) {
    if (size == 1) return data[0];
    int const stride = size / 2;
    for (int i = 0; i < stride; i++) {
        data[i] += data[i+stride];
    }

    return recursiveReduce(data, stride);
}

int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    cudaSetDevice(dev);

    bool bResult = false;

    int size = 1 << 24;
    printf("    with array size %d ", size);

    int blocksize = 512;
    if (argc > 1) {
        blocksize = atoi(argv[1]);
    }

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x);
    printf("grid %d block %d\n", grid.x, block.x);

    size_t bytes = size * sizeof(int);
    int *h_idata = (int*)malloc(bytes);
    int *h_odata = (int*)malloc(grid.x * sizeof(int));
    int *tmp = (int*)malloc(bytes);

    for (int i = 0; i < size; i++) {
        //h_idata[i] = (int)(rand() & 0xFF);
        h_idata[i] = 2;
    }

    memcpy(tmp, h_idata, bytes);

    double iStart, iElaps;
    int gpu_sum = 0;

    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc(&d_idata, bytes));
    CHECK(cudaMalloc(&d_odata, grid.x * sizeof(int)));

    iStart = cpuSecond();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = cpuSecond() - iStart;
    printf("cpu reduce elapsed: %f s, cpu_sum: %d\n", iElaps, cpu_sum);

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    warmup<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i<grid.x; ++i) gpu_sum += h_odata[i];
    printf("gpu warmup elapsed %f s, gpu_sum %d <<<grid %d block %d>>>\n",
            iElaps, gpu_sum, grid.x, block.x);
        
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i<grid.x; ++i) gpu_sum += h_odata[i];
    printf("gpu Neighbored elapsed %f s, gpu_sum %d <<<grid %d block %d>>>\n",
            iElaps, gpu_sum, grid.x, block.x);

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i<grid.x; ++i) gpu_sum += h_odata[i];
    printf("gpu NeighboredLess elapsed %f s, gpu_sum %d <<<grid %d block %d>>>\n",
            iElaps, gpu_sum, grid.x, block.x);

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceInterLeaved<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i<grid.x; ++i) gpu_sum += h_odata[i];
    printf("gpu Interleaved elapsed %f s, gpu_sum %d <<<grid %d block %d>>>\n",
            iElaps, gpu_sum, grid.x, block.x);

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceUnrolling2<<<grid.x / 2, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i<grid.x/2; i++) gpu_sum += h_odata[i];
    printf("gpu Unrolling2 elapsed %f s, gpu_sum %d <<< grid %d block %d>>>\n",
            iElaps, gpu_sum, grid.x/2, block.x);


    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceUnrolling8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i<grid.x/8; i++) gpu_sum += h_odata[i];
    printf("gpu Unrolling8 elapsed %f s, gpu_sum %d <<< grid %d block %d>>>\n",
            iElaps, gpu_sum, grid.x/2, block.x);

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i<grid.x/8; i++) gpu_sum += h_odata[i];
    printf("gpu UnrollWarps8 elapsed %f s, gpu_sum %d <<< grid %d block %d>>>\n",
            iElaps, gpu_sum, grid.x/2, block.x);


    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceCompleteUnrollWarps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i<grid.x/8; i++) gpu_sum += h_odata[i];
    printf("gpu CompleteUnrollWarps8 elapsed %f s, gpu_sum %d <<< grid %d block %d>>>\n",
            iElaps, gpu_sum, grid.x/2, block.x);

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    switch (blocksize) {
        case 1024:
            reduceCompleteUnroll<1024><<<grid.x / 8, block>>>(d_idata, d_odata, size);
            break;
        case 512:
            reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_idata, d_odata, size);
            break;
         case 256:
            reduceCompleteUnroll<256><<<grid.x / 8, block>>>(d_idata, d_odata, size);
            break;
         case 128:
            reduceCompleteUnroll<128><<<grid.x / 8, block>>>(d_idata, d_odata, size);
            break;
         case 64:
            reduceCompleteUnroll<64><<<grid.x / 8, block>>>(d_idata, d_odata, size);
            break;
         case 32:
            reduceCompleteUnroll<32><<<grid.x / 8, block>>>(d_idata, d_odata, size);
            break;
    }
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i=0; i<grid.x/8; i++) gpu_sum += h_odata[i];
    printf("gpu CompleteUnroll elapsed %f s, gpu_sum %d <<< grid %d block %d>>>\n",
            iElaps, gpu_sum, grid.x/2, block.x);


    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);
    
    cudaDeviceReset();
    bResult = (gpu_sum == cpu_sum);
    if (!bResult) printf("Test Failed\n");
    return EXIT_SUCCESS;
}