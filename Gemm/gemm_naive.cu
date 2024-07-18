#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

#include "utils.cuh"

__global__ void naiveSgemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        float psum = 0.0;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = psum;
    }
}

int main(void) {
    const int BM = 16, BN = 16;
    const int TEST_M = 1024, TEST_N = 1024, TEST_K = 1024;
    dim3 blockDim_T(BN, BM);
    dim3 gridDim_T((TEST_N + BN - 1) / BN, (TEST_M + BM - 1) / BM);

    void (*gpuSgemm) (float*, float*, float*, const int, const int, const int) = naiveSgemm;
    float max_error = testError(gpuSgemm, gridDim_T, blockDim_T, TEST_M, TEST_N, TEST_K);
    printf("Max error: %f\n", max_error);

    printf("\n Kernel = naiveSgemm\n");
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
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
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
