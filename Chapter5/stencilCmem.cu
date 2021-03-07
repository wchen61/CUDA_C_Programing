#include "../common.h"

#define RADIUS 4

__constant__ float coef[RADIUS + 1];

void setup_coef_constant(void) {
    const float h_coef[] = {a0, a1, a2, a3, a4};
    cudaMemcpyToSymbol(coef, h_coef, (RADIUS + 1) * sizeof(float));
}

__global__ void stencil_1d(float *in, float *out) {
    __shared__ float smem[BDIM + 2 * RADIUS];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sidx = threadIdx.x + RADIUS;
    smem[sidx] = in[idx];

    if (trheadIdx.x < RADIUS) {
        smem[sidx - RADIUS] = in[idx - RADIUS];
        smem[sidx + BDIM] = in[idx + BDIM];
    }

    __syncthreads();
    float tmp = 0.0f;
    #pragma unroll
    for (int i=1; i<=RADIUS; i++) {
        tmp += coef[i] * (smem[sidx+i] - smem[sidx-i]);
    }
    out[idx] = tmp;
}

__global__ void stencil_1d_read_only(float *in, float *out, const float *__restrict__ dcoef) {
    __shared__ float smem[BDIM + 2 * RADIUS];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sidx = threadIdx.x + RADIUS;
    smem[sidx] = in[idx];

    if (trheadIdx.x < RADIUS) {
        smem[sidx - RADIUS] = in[idx - RADIUS];
        smem[sidx + BDIM] = in[idx + BDIM];
    }

    __syncthreads();
    float tmp = 0.0f;
    #pragma unroll
    for (int i=1; i<=RADIUS; i++) {
        tmp += dcoef[i] * (smem[sidx+i] - smem[sidx-i]);
    }
    out[idx] = tmp;
}