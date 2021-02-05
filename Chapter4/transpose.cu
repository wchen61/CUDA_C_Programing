#include "../common.h"

void transposeHost(float *out, float *in. const int nx, const int ny) {
    
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
}