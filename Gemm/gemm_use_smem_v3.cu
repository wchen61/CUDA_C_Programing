#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

#include "utils.cuh"

// BM = blockDim.y * TM
// BN = blockDim.x * TN
template <int BM, int BN, int BK, int TM, int TN, int BLOCK_SIZE>
__global__ void gemm_use_smem(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    // Top-left element index by this thread
    unsigned int m = blockIdx.y * BM + threadIdx.y * TM;
    unsigned int n = blockIdx.x * BN + threadIdx.x * TN;

    if (m >= M || n >= N)
        return;

    __shared__ float tileA[BK][BM];
    __shared__ float tileB[BK][BN];
    float r_c[TM][TN] = {0.0};
    float r_a[TM] = {0.0};
    float r_b[TN] = {0.0};
 
    const int tileA_element = (BM * BK) / BLOCK_SIZE;
    const int tileB_element = (BK * BN) / BLOCK_SIZE;

    for (int k = 0; k < K; k += BK) {
        #pragma unroll
        for (int i = 0; i < tileA_element; i += 4) {
            int tileA_m = (tid * tileA_element + i) / BK;
            int tileA_k = (tid * tileA_element + i) % BK;
            int gmem_m = by * BM + tileA_m;
            float4 v = *reinterpret_cast<const float4*>(a + OFFSET(gmem_m, k + tileA_k, K));
            tileA[tileA_k    ][tileA_m] = v.x;
            tileA[tileA_k + 1][tileA_m] = v.y;
            tileA[tileA_k + 2][tileA_m] = v.z;
            tileA[tileA_k + 3][tileA_m] = v.w;
        }

        #pragma unroll
        for (int i = 0; i < tileB_element; i += 4) {
            int tileB_k = (tid * tileB_element + i) / BN;
            int tileB_n = (tid * tileB_element + i) % BN;
            int gmem_n = bx * BN + tileB_n;
            *reinterpret_cast<float4*>(&tileB[tileB_k][tileB_n]) = *reinterpret_cast<const float4*>(b + OFFSET(k + tileB_k, gmem_n, N));
        }
        __syncthreads();
        
        /*if (threadIdx.x == 0 && threadIdx.y == 0) {
            printf("\nK: %d\n", k);
            printf("\ntileA:");
            for (int m = 0; m < BM; m++) { 
                for (int k = 0; k < BK; k++) {
                    printf(" %.3f,", tileA[m][k]);
                }
                printf("\n");
            }
            printf("\ntileB:");
            for (int k = 0; k < BK; k++) { 
                for (int n = 0; n < BN; n++) {
                    printf(" %.3f,", tileB[k][n]);
                }
                printf("\n");
            }
        }*/

        for (int tk = 0; tk < BK; tk++) {
            #pragma unroll
            for (int tm = 0; tm < TM; tm += 4) {
                int interval = (tm < TM / 2) ? tm : tm - TM / 2 +  BM / 2;
                int a_smem_m = ty * TM / 2 + interval;
                *reinterpret_cast<float4*>(&r_a[tm]) = *reinterpret_cast<float4*>(&tileA[tk][a_smem_m]);
            }
            #pragma unroll
            for (int tn = 0; tn < TN; tn +=4) {
                int interval = (tn < TN / 2) ? tn : tn - TN / 2 + BN / 2;
                int b_smem_n = tx * TN / 2 + interval;
                *reinterpret_cast<float4*>(&r_b[tn]) = *reinterpret_cast<float4*>(&tileB[tk][b_smem_n]);
            }

            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_a[tm] * r_b[tn];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int interval_M = (i < TM / 2) ? i : i - TM / 2 + BM / 2;
        int c_gmem_m = by * BM + ty * TM / 2 + interval_M;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int interval_N = (j < TN / 2) ? j : j - TN / 2 + BN / 2;
            int c_gmem_n = bx * BN + tx * TN / 2 + interval_N;
            *reinterpret_cast<float4*>(c + OFFSET(c_gmem_m, c_gmem_n, N)) = *reinterpret_cast<float4*>(&r_c[i][j]);
        }
    }
}

int main(void) {
    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    const int TEST_M = 1024, TEST_N = 1024, TEST_K = 1024;
    const int BLOCK_SIZE = (BN / TN) * (BM / TM);
    dim3 blockDim_T(BN / TN, BM / TM);
    dim3 gridDim_T(TEST_N / BN, TEST_M / BM);

    void (*gpuSgemm) (float*, float*, float*, const int, const int, const int) = gemm_use_smem<BM, BN, BK, TM, TN, BLOCK_SIZE>;
    float max_error = testError(gpuSgemm, gridDim_T, blockDim_T, TEST_M, TEST_N, TEST_K);
    printf("Max error: %f\n", max_error);
    printf("\n Kernel = gemm_use_smem\n");
    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};

    const int outer_repeat = 10, inner_repeat = 1;
    const int TESTNUM = 15;
    for (int i = 0; i < TESTNUM; i++) {
        int M = M_list[i];
        int N = N_list[i];
        int K = K_list[i];
        dim3 blockDim(BN / TN, BM / TM);
        dim3 gridDim((N + BN - 1) / BN, (M  + BM - 1) / BM);
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
