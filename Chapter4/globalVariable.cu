#include <cuda_runtime.h>
#include <stdio.h>

__device__ float devData;

__global__ void checkGlobalVariable() {
    printf("Device: the value of the global variable is %f\n", devData);
    devData += 2.0f;
}

int main(void) {
    float value = 3.14f;
    cudaMemcpyToSymbol(devData, &value, sizeof(float));
    printf("Host: cpoed %f to the global variable\n", value);

    checkGlobalVariable<<<1, 1>>>();

    //cudaMemcpyFromSymbol(&value, devData, sizeof(float));

    float *dptr = NULL;
    cudaGetSymbolAddress((void**)&dptr, devData);
    cudaMemcpy(&value, dptr, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Host: the value changed by the kernel to %f\n", value);
    
    cudaDeviceReset();
    return EXIT_SUCCESS;
}