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

    __shared__ float tileA[2][BK][BM];
    __shared__ float tileB[2][BK][BN];
    float r_c[TM][TN] = {0.0};
    float r_a[2][TM] = {0.0};
    float r_b[2][TN] = {0.0};
 
    const int tileA_element = (BM * BK) / BLOCK_SIZE;
    const int tileB_element = (BK * BN) / BLOCK_SIZE;

    float4 r_a_load[tileA_element / 4];
    float4 r_b_load[tileB_element / 4];
    {
        #pragma unroll
        for (int i = 0; i < tileA_element; i += 4) {
            int tileA_m = (tid * tileA_element + i) / BK;
            int tileA_k = (tid * tileA_element + i) % BK;
            int gmem_m = by * BM + tileA_m;
            float4 v = *reinterpret_cast<const float4*>(a + OFFSET(gmem_m, tileA_k, K));
            tileA[0][tileA_k    ][tileA_m] = v.x;
            tileA[0][tileA_k + 1][tileA_m] = v.y;
            tileA[0][tileA_k + 2][tileA_m] = v.z;
            tileA[0][tileA_k + 3][tileA_m] = v.w;
        }

        #pragma unroll
        for (int i = 0; i < tileB_element; i += 4) {
            int tileB_k = (tid * tileB_element + i) / BN;
            int tileB_n = (tid * tileB_element + i) % BN;
            int gmem_n = bx * BN + tileB_n;
            *reinterpret_cast<float4*>(&tileB[0][tileB_k][tileB_n]) = *reinterpret_cast<const float4*>(b + OFFSET(tileB_k, gmem_n, N));
        }
    }

    int smem_sel = 0;
    int smem_sel_next = 1;
    __syncthreads();

    for (int k = BK; k < K; k += BK) {
        #pragma unroll
        for (int i = 0; i < tileA_element; i += 4) {
            int tileA_m = (tid * tileA_element + i) / BK;
            int tileA_k = (tid * tileA_element + i) % BK;
            int gmem_m = by * BM + tileA_m;
            r_a_load[i / 4] = *reinterpret_cast<const float4*>(a + OFFSET(gmem_m, k + tileA_k, K));
        }

        #pragma unroll
        for (int i = 0; i < tileB_element; i += 4) {
            int tileB_k = (tid * tileB_element + i) / BN;
            int tileB_n = (tid * tileB_element + i) % BN;
            int gmem_n = bx * BN + tileB_n;
            r_b_load[i / 4] = *reinterpret_cast<const float4*>(b + OFFSET(k + tileB_k, gmem_n, N));
        }
        //__syncthreads();
        
        /*if (threadIdx.x == 0 && threadIdx.y == 0) {
            printf("\nK: %d\n", k);
            printf("\ntileA:");
            for (int m = 0; m < BM; m++) { 
                for (int k = 0; k < BK; k++) {
                    printf(" %.3f,", tileA[smem_sel][m][k]);
                }
                printf("\n");
            }
            printf("\ntileB:");
            for (int k = 0; k < BK; k++) { 
                for (int n = 0; n < BN; n++) {
                    printf(" %.3f,", tileB[smem_sel][k][n]);
                }
                printf("\n");
            }
        }*/

        #pragma unroll
        for (int tm = 0; tm < TM; tm += 4) {
            int interval = (tm < TM / 2) ? tm : tm - TM / 2 +  BM / 2;
            int a_smem_m = ty * TM / 2 + interval;
            *reinterpret_cast<float4*>(&r_a[0][tm]) = *reinterpret_cast<float4*>(&tileA[smem_sel][0][a_smem_m]);
        }
        #pragma unroll
        for (int tn = 0; tn < TN; tn +=4) {
            int interval = (tn < TN / 2) ? tn : tn - TN / 2 + BN / 2;
            int b_smem_n = tx * TN / 2 + interval;
            *reinterpret_cast<float4*>(&r_b[0][tn]) = *reinterpret_cast<float4*>(&tileB[smem_sel][0][b_smem_n]);
        }
        int reg_sel = 0;
        int reg_sel_next = 1;

        #pragma unroll
        for (int tk = 1; tk < BK; tk++) {
            #pragma unroll
            for (int tm = 0; tm < TM; tm += 4) {
                int interval = (tm < TM / 2) ? tm : tm - TM / 2 +  BM / 2;
                int a_smem_m = ty * TM / 2 + interval;
                *reinterpret_cast<float4*>(&r_a[reg_sel_next][tm]) = *reinterpret_cast<float4*>(&tileA[smem_sel][tk][a_smem_m]);
            }
            #pragma unroll
            for (int tn = 0; tn < TN; tn +=4) {
                int interval = (tn < TN / 2) ? tn : tn - TN / 2 + BN / 2;
                int b_smem_n = tx * TN / 2 + interval;
                *reinterpret_cast<float4*>(&r_b[reg_sel_next][tn]) = *reinterpret_cast<float4*>(&tileB[smem_sel][tk][b_smem_n]);
            }

            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_a[reg_sel][tm] * r_b[reg_sel][tn];
                }
            }
            reg_sel = !reg_sel;
            reg_sel_next = !reg_sel_next;
        }
        #pragma unroll
        for (int tm = 0; tm < TM; tm++) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] += r_a[reg_sel][tm] * r_b[reg_sel][tn];
            }
        }

        #pragma unroll
        for (int i = 0; i < tileA_element; i += 4) {
            int tileA_m = (tid * tileA_element + i) / BK;
            int tileA_k = (tid * tileA_element + i) % BK;
            float4 v = r_a_load[i/4];
            tileA[smem_sel_next][tileA_k    ][tileA_m] = v.x;
            tileA[smem_sel_next][tileA_k + 1][tileA_m] = v.y;
            tileA[smem_sel_next][tileA_k + 2][tileA_m] = v.z;
            tileA[smem_sel_next][tileA_k + 3][tileA_m] = v.w;
        }

        #pragma unroll
        for (int i = 0; i < tileB_element; i += 4) {
            int tileB_k = (tid * tileB_element + i) / BN;
            int tileB_n = (tid * tileB_element + i) % BN;
            *reinterpret_cast<float4*>(&tileB[smem_sel_next][tileB_k][tileB_n]) = r_b_load[i / 4];
        }

        __syncthreads();
        smem_sel = !smem_sel;
        smem_sel_next = !smem_sel_next;
    }

    #pragma unroll
    for (int tm = 0; tm < TM; tm += 4) {
        int interval = (tm < TM / 2) ? tm : tm - TM / 2 +  BM / 2;
        int a_smem_m = ty * TM / 2 + interval;
        *reinterpret_cast<float4*>(&r_a[0][tm]) = *reinterpret_cast<float4*>(&tileA[smem_sel][0][a_smem_m]);
    }
    #pragma unroll
    for (int tn = 0; tn < TN; tn +=4) {
        int interval = (tn < TN / 2) ? tn : tn - TN / 2 + BN / 2;
        int b_smem_n = tx * TN / 2 + interval;
        *reinterpret_cast<float4*>(&r_b[0][tn]) = *reinterpret_cast<float4*>(&tileB[smem_sel][0][b_smem_n]);
    }
    int reg_sel = 0;
    int reg_sel_next = 1;

    #pragma unroll
    for (int tk = 1; tk < BK; tk++) {
        #pragma unroll
        for (int tm = 0; tm < TM; tm += 4) {
            int interval = (tm < TM / 2) ? tm : tm - TM / 2 +  BM / 2;
            int a_smem_m = ty * TM / 2 + interval;
            *reinterpret_cast<float4*>(&r_a[reg_sel_next][tm]) = *reinterpret_cast<float4*>(&tileA[smem_sel][tk][a_smem_m]);
        }
        #pragma unroll
        for (int tn = 0; tn < TN; tn +=4) {
            int interval = (tn < TN / 2) ? tn : tn - TN / 2 + BN / 2;
            int b_smem_n = tx * TN / 2 + interval;
            *reinterpret_cast<float4*>(&r_b[reg_sel_next][tn]) = *reinterpret_cast<float4*>(&tileB[smem_sel][tk][b_smem_n]);
        }

        #pragma unroll
        for (int tm = 0; tm < TM; tm++) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] += r_a[reg_sel][tm] * r_b[reg_sel][tn];
            }
        }
        reg_sel = !reg_sel;
        reg_sel_next = !reg_sel_next;
    }
    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
            r_c[tm][tn] += r_a[reg_sel][tm] * r_b[reg_sel][tn];
        }
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