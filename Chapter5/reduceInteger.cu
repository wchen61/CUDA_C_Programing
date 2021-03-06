#include "../common.h"

#define DIM 128

__global__ void reduceGmem(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n) return;

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

__global__ void reduceSmem(int *g_idata, int *g_odata, unsigned int n) {
    __shared__ int smem[DIM];
    unsigned int tid = threadIdx.x;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > n) return;

    int *idata = g_idata + blockIdx.x * blockDim.x;
    smem[tid] = idata[tid];
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid+512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid+256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid+128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid+64];
    __syncthreads();

    if (tid < 32) {
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}



__global__ void reduceSmemUnroll(int *g_idata, int *g_odata, unsigned int n) {
    __shared__ int smem[DIM];
    unsigned int tid = threadIdx.x;

    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    int tmpSum = 0;

    if (idx + 3 * blockDim.x <= n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        tmpSum = a1 + a2 + a3 + a4;
    }

    smem[tid] = tmpSum;
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid+512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid+256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid+128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid+64];
    __syncthreads();

    if (tid < 32) {
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceSmemUnrollDyn(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int smem[];
    unsigned int tid = threadIdx.x;

    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    int tmpSum = 0;

    if (idx + 3 * blockDim.x <= n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        tmpSum = a1 + a2 + a3 + a4;
    }

    smem[tid] = tmpSum;
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid+512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid+256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid+128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid+64];
    __syncthreads();

    if (tid < 32) {
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

int recursiveReduce(int *data, int const size) {
    if (size == 1) return data[0];
    int const stride = size / 2;
    for (int i = 0; i < stride; i++) {
        data[i] += data[i+stride];
    }

    return recursiveReduce(data, stride);
}


int main(int argc, char** argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    cudaSetDevice(dev);

    int size = 1 << 20;
    printf("    with array size %d ", size);

    int blocksize = DIM;
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
        srand((unsigned)time(NULL));
        h_idata[i] = (int)(rand() % 0xFF);
        //h_idata[i] = 2;
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

    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = cpuSecond();
    reduceGmem<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i=0; i<grid.x; i++) {
        gpu_sum += h_odata[i];
    }
    printf("gpu reduce Gmem elapsed: %f s, gpu_sum: %d\n", iElaps, gpu_sum);

    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = cpuSecond();
    reduceSmem<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i=0; i<grid.x; i++) {
        gpu_sum += h_odata[i];
    }
    printf("gpu reduce Smem elapsed: %f s, gpu_sum: %d\n", iElaps, gpu_sum);

    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = cpuSecond();
    reduceSmemUnroll<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i=0; i<grid.x; i++) {
        gpu_sum += h_odata[i];
    }
    printf("gpu reduce Smem Unroll elapsed: %f s, gpu_sum: %d\n", iElaps, gpu_sum);

    
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = cpuSecond();
    reduceSmemUnrollDyn<<<grid.x / 4, block, DIM * sizeof(int)>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i=0; i<grid.x; i++) {
        gpu_sum += h_odata[i];
    }
    printf("gpu reduce Smem Unroll Dynamic elapsed: %f s, gpu_sum: %d\n", iElaps, gpu_sum);
}