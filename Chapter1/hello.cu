#include <stdio.h>


__global__ void helloFromGPU(void) {
    printf("Hello World from GPU, blockIdx: %d threadIdx: %d\n", blockIdx.x, threadIdx.x);
}

int main(void) {
    printf("Hello World from CPU1\n");
    helloFromGPU<<<1024, 10>>>();
    //cudaDeviceSynchronize();
    printf("Hello World from CPU2\n");
    cudaDeviceReset();
    return 0;
}