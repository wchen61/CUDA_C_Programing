#include <cuda_runtime.h>
#include <stdio.h>
#include "../common.h"

int main(int argc, char **argv) {
    int dev = 0;
    cudaSetDevice(dev);

    unsigned int isize = 1 << 22;
    unsigned int nbytes = isize * sizeof(float);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting at ", argv[0]);
    printf("device %d: %s memory size %d nbytes %5.2fMB\n", dev, deviceProp.name, isize, nbytes / (1024.0f * 1024.0f));

    float *h_a = (float*)malloc(nbytes);
    float *d_a;
    CHECK(cudaMalloc((float**)&d_a, nbytes));

    for (unsigned int i=0; i<isize; i++) h_a[i] = 0.5f;

    cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);

    cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    free(h_a);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}