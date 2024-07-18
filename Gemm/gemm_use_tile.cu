#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

#include "utils.cuh"

__global__ void gemm_use_tile(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    // Element number process by thread
    const unsigned kCount = 4;

    // Iteration Count on A
    const unsigned int iterationA = 2;
    
    // Iteration Count on B 
    const unsigned int iterationB = 2;

    // The shape processed by ThreadBlock, 
    unsigned tile_M = blockDim.y * kCount * iterationA;
    unsigned tile_N = blockDim.x * kCount * iterationB;

    // interleave of A and B
    unsigned intervalA = tile_M / iterationA;
    unsigned intervalB = tile_N / iterationB;

    // Top-left element index by this thread
    unsigned int m = blockIdx.y * tile_M + threadIdx.y * kCount;
    unsigned int n = blockIdx.x * tile_N + threadIdx.x * kCount;

    if (m >= M || n >= N)
        return;

    float4 r_a;
    float4 r_b;
    float4 r_c[iterationA][iterationB][4];
    memset(r_c, 0, sizeof(r_c));

    for (int k = 0; k < K; k++) {
#pragma unroll
        for (unsigned iterA = 0; iterA < iterationA; ++iterA) {
            r_a.x = a[OFFSET(m + iterA * intervalA    , k, K)];
            r_a.y = a[OFFSET(m + iterA * intervalA + 1, k, K)];
            r_a.z = a[OFFSET(m + iterA * intervalA+ 2, k, K)];
            r_a.w = a[OFFSET(m + iterA * intervalA+ 3, k, K)];
#pragma unroll
            for (unsigned iterB = 0; iterB < iterationB; ++iterB) {
                r_b = *reinterpret_cast<const float4*>(b + OFFSET(k, n + iterB * intervalB, N));
                r_c[iterA][iterB][0].x += r_a.x * r_b.x;
                r_c[iterA][iterB][0].y += r_a.x * r_b.y;
                r_c[iterA][iterB][0].z += r_a.x * r_b.z;
                r_c[iterA][iterB][0].w += r_a.x * r_b.w;
 
                r_c[iterA][iterB][1].x += r_a.y * r_b.x;
                r_c[iterA][iterB][1].y += r_a.y * r_b.y;
                r_c[iterA][iterB][1].z += r_a.y * r_b.z;
                r_c[iterA][iterB][1].w += r_a.y * r_b.w;

                r_c[iterA][iterB][2].x += r_a.z * r_b.x;
                r_c[iterA][iterB][2].y += r_a.z * r_b.y;
                r_c[iterA][iterB][2].z += r_a.z * r_b.z;
                r_c[iterA][iterB][2].w += r_a.z * r_b.w;

                r_c[iterA][iterB][3].x += r_a.w * r_b.x;
                r_c[iterA][iterB][3].y += r_a.w * r_b.y;
                r_c[iterA][iterB][3].z += r_a.w * r_b.z;
                r_c[iterA][iterB][3].w += r_a.w * r_b.w;           
            }
        }
    }

#pragma unroll
    for (unsigned iterA = 0; iterA < iterationA; ++iterA) {
        for (unsigned iterB = 0; iterB < iterationB; ++iterB) {
            for (unsigned i = 0; i < kCount; i++) {
                *reinterpret_cast<float4*>(c + OFFSET(m + i + iterA * intervalA, n + iterB * intervalB, N)) = r_c[iterA][iterB][i]; 
            }
        }
    }
}	

int main(void) {
    const int BM = 16, BN = 16;
    const int TEST_M = 1024, TEST_N = 1024, TEST_K = 1024;
    //const int BM = 1, BN = 1;
    //const int TEST_M = 16, TEST_N = 16, TEST_K = 16;
    dim3 blockDim_T(BN, BM);
    dim3 gridDim_T((TEST_N / 8 + BN - 1) / BN, (TEST_M / 8 + BM - 1) / BM);

    void (*gpuSgemm) (float*, float*, float*, const int, const int, const int) = gemm_use_tile;
    float max_error = testError(gpuSgemm, gridDim_T, blockDim_T, TEST_M, TEST_N, TEST_K);
    printf("Max error: %f\n", max_error);

    printf("\n Kernel = gemm_use_128\n");
    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};

    const int outer_repeat = 10, inner_repeat = 1;
    const int TESTNUM = 15;
    for (int i = 0; i < TESTNUM; i++) {
        int M = M_list[i];
        int N = N_list[i];
        int K = K_list[i];
        dim3 blockDim(BN, BM);
        dim3 gridDim((N / 4 + BN - 1) / BN, (M / 4 + BM - 1) / BM);
        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j++) {
            double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;
        printf("M N K = %6d %6d %6d, avg_sec = %f, max_sec = %f, min_sec = %f, avg_Gflops = %f\n", M, N, K, avg_sec, max_sec, min_sec, avg_Gflops);
    }
    return 0;
}
