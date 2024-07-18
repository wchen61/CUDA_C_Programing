#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

#include "utils.cuh"

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void gemm_use_smem(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    // Element number process by thread
    const unsigned kCount = 4;

    // Iteration Count on M 
    const unsigned int iterationM = 2;
    
    // Iteration Count on N
    const unsigned int iterationN = 2;

    // Iteration Count on K
    const unsigned int iterationK = 1;

    // The shape processed by ThreadBlock, 
    const unsigned tile_M = BLOCK_DIM_Y * kCount * iterationM;
    const unsigned tile_N = BLOCK_DIM_X * kCount * iterationN;

    // Assume that blockDim.x == blockDim.y
    const unsigned tile_K = BLOCK_DIM_X * kCount * iterationK;
    
    // interleave of M, N, K
    const unsigned intervalM = tile_M / iterationM;
    const unsigned intervalN = tile_N / iterationN;
    const unsigned intervalK = tile_K / iterationK;

    // Top-left element index by this thread
    unsigned int m = blockIdx.y * tile_M + threadIdx.y * kCount;
    unsigned int n = blockIdx.x * tile_N + threadIdx.x * kCount;

    if (m >= M || n >= N)
        return;

    __shared__ float4 tileA[tile_M][tile_K / kCount];
    __shared__ float4 tileB[tile_K][tile_N / kCount];
    float4 bufferA[iterationM][iterationK][kCount];
    float4 bufferB[iterationK][iterationN][kCount];

    float4 r_c[iterationM][iterationN][4];
    memset(r_c, 0, sizeof(r_c));

    for (int k = 0; k < K; k += tile_K) {
#pragma unroll
        for (unsigned i = 0; i < iterationM; i++) {
#pragma unroll
            for (unsigned j = 0; j < iterationK; j++) {
                bufferA[i][j][0] = *reinterpret_cast<const float4*>(a + OFFSET(m + i * intervalM    , k + threadIdx.x * kCount, K));
                bufferA[i][j][1] = *reinterpret_cast<const float4*>(a + OFFSET(m + i * intervalM + 1, k + threadIdx.x * kCount, K));
                bufferA[i][j][2] = *reinterpret_cast<const float4*>(a + OFFSET(m + i * intervalM + 2, k + threadIdx.x * kCount, K));
                bufferA[i][j][3] = *reinterpret_cast<const float4*>(a + OFFSET(m + i * intervalM + 3, k + threadIdx.x * kCount, K));
                //printf("gmem->register (%d, %d), %d, %d: %.3f, %.3f, %.3f, %.3f\n", 
                //    threadIdx.y, threadIdx.x, i, j, bufferA[i][j][0].x, bufferA[i][j][0].y, bufferA[i][j][0].z, bufferA[i][j][0].w);
            }
        }

#pragma unroll
        for (unsigned i = 0; i < iterationK; i++) {
#pragma unroll
            for (unsigned j = 0; j < iterationN; j++) {
                bufferB[i][j][0] = *reinterpret_cast<const float4*>(b + OFFSET(k + threadIdx.y * 4   , n + j * intervalN, N));
                bufferB[i][j][1] = *reinterpret_cast<const float4*>(b + OFFSET(k + threadIdx.y * 4 + 1, n + j * intervalN, N));
                bufferB[i][j][2] = *reinterpret_cast<const float4*>(b + OFFSET(k + threadIdx.y * 4 + 2, n + j * intervalN, N));
                bufferB[i][j][3] = *reinterpret_cast<const float4*>(b + OFFSET(k + threadIdx.y * 4 + 3, n + j * intervalN, N));
                //if(threadIdx.x == 1 && threadIdx.y == 1)
                //    printf("gmem->register, %d, %d: %.3f, %.3f, %.3f, %.3f\n", i, j, bufferB[i][j][0].x, bufferB[i][j][0].y, bufferB[i][j][0].z, bufferB[i][j][0].w);
            }
        }
        __syncthreads();

#pragma unroll
        for (unsigned i = 0; i < iterationM; i++) {
#pragma unroll
            for (unsigned j = 0; j < iterationK; j++) {
                tileA[threadIdx.y * kCount + i * intervalM    ][threadIdx.x + j * intervalK] = bufferA[i][j][0];
                tileA[threadIdx.y * kCount + i * intervalM + 1][threadIdx.x + j * intervalK] = bufferA[i][j][1];
                tileA[threadIdx.y * kCount + i * intervalM + 2][threadIdx.x + j * intervalK] = bufferA[i][j][2];
                tileA[threadIdx.y * kCount + i * intervalM + 3][threadIdx.x + j * intervalK] = bufferA[i][j][3];
            }
        }

#pragma unroll
        for (unsigned i = 0; i < iterationK; i++) {
#pragma unroll
            for (unsigned j = 0; j < iterationN; j++) {
                unsigned row = threadIdx.y * kCount + i * intervalK;
                unsigned col = (threadIdx.x * kCount + j * intervalN) / 4;
                //printf("register->smem, %d, %d: %.3f, %.3f, %.3f, %.3f\n", i, j, bufferB[i][j][0].x, bufferB[i][j][0].y, bufferB[i][j][0].z, bufferB[i][j][0].w);
                tileB[row    ][col] = bufferB[i][j][0];
                tileB[row + 1][col] = bufferB[i][j][1];
                tileB[row + 2][col] = bufferB[i][j][2];
                tileB[row + 3][col] = bufferB[i][j][3];
                //if(threadIdx.x == 1 && threadIdx.y == 1)
                //    printf("%d, %d: %.3f, %.3f, %.3f, %.3f\n", row, col, tileB[row][col].x, tileB[row][col].y, tileB[row][col].z, tileB[row][col].w);
            }
        }
        __syncthreads();

        /*if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("K: %d\n", k);
        for (int m = 0; m < tile_M; m++) { 
            for (int k = 0; k < tile_K / kCount; k++) {
                printf("tileA %d, %d: %.3f, %.3f, %.3f, %.3f\n", m, k, tileA[m][k].x, tileA[m][k].y, tileA[m][k].z, tileA[m][k].w);
            }
        }
        for (int k = 0; k < tile_K; k++) { 
            for (int n = 0; n < tile_N / kCount; n++) {
                printf("tileB %d, %d: %.3f, %.3f, %.3f, %.3f\n", 
                    k, n, tileB[k][n].x, tileB[k][n].y, tileB[k][n].z, tileB[k][n].w);
            }
        }
        }*/
        float4 r_a[4];
        float4 r_b[4];
        for (unsigned i = 0; i < tile_K / 4; i++) {
            for (unsigned iterM = 0; iterM < iterationM; ++iterM) {
                unsigned mInTileA = threadIdx.y * kCount + iterM * intervalM;
                r_a[0].x = tileA[mInTileA    ][i].x;
                r_a[0].y = tileA[mInTileA + 1][i].x;
                r_a[0].z = tileA[mInTileA + 2][i].x;
                r_a[0].w = tileA[mInTileA + 3][i].x;
                r_a[1].x = tileA[mInTileA    ][i].y;
                r_a[1].y = tileA[mInTileA + 1][i].y;
                r_a[1].z = tileA[mInTileA + 2][i].y;
                r_a[1].w = tileA[mInTileA + 3][i].y;
                r_a[2].x = tileA[mInTileA    ][i].z;
                r_a[2].y = tileA[mInTileA + 1][i].z;
                r_a[2].z = tileA[mInTileA + 2][i].z;
                r_a[2].w = tileA[mInTileA + 3][i].z;
                r_a[3].x = tileA[mInTileA    ][i].w;
                r_a[3].y = tileA[mInTileA + 1][i].w;
                r_a[3].z = tileA[mInTileA + 2][i].w;
                r_a[3].w = tileA[mInTileA + 3][i].w;
                for (unsigned iterN = 0; iterN < iterationN; ++iterN) {
                    unsigned nInTileB = (threadIdx.x * kCount + iterN * intervalN) / 4;
                    r_b[0] = tileB[i*4    ][nInTileB];
                    r_b[1] = tileB[i*4 + 1][nInTileB];
                    r_b[2] = tileB[i*4 + 2][nInTileB];
                    r_b[3] = tileB[i*4 + 3][nInTileB];
                    /*if (iterM == 0 && iterN == 0 && threadIdx.x == 0 && threadIdx.y == 0)
                         printf("(k:%d, i: %d) r_a 0,  %.3f, %.3f, %.3f, %.3f x %.3f, %.3f %.3f %.3f\n", 
                                k, i, r_a[0].x, r_a[1].x, r_a[2].x, r_a[3].x,
                                r_b[0].x, r_b[1].x, r_b[2].x, r_b[3].x);*/
                    for (int e = 0; e < kCount; e++) {
                        float4 a = r_a[e];
                        float4 b = r_b[e];
                        r_c[iterM][iterN][0].x += a.x * b.x;
                        r_c[iterM][iterN][0].y += a.x * b.y;
                        r_c[iterM][iterN][0].z += a.x * b.z;
                        r_c[iterM][iterN][0].w += a.x * b.w;
 
                        r_c[iterM][iterN][1].x += a.y * b.x;
                        r_c[iterM][iterN][1].y += a.y * b.y;
                        r_c[iterM][iterN][1].z += a.y * b.z;
                        r_c[iterM][iterN][1].w += a.y * b.w;

                        r_c[iterM][iterN][2].x += a.z * b.x;
                        r_c[iterM][iterN][2].y += a.z * b.y;
                        r_c[iterM][iterN][2].z += a.z * b.z;
                        r_c[iterM][iterN][2].w += a.z * b.w;

                        r_c[iterM][iterN][3].x += a.w * b.x;
                        r_c[iterM][iterN][3].y += a.w * b.y;
                        r_c[iterM][iterN][3].z += a.w * b.z;
                        r_c[iterM][iterN][3].w += a.w * b.w;           
                    }
                    //printf("%d, %d: %f\n", iterM, iterN, r_c[iterM][iterN][0].x);
                }
            }
        }
    }

#pragma unroll
    for (unsigned iterM = 0; iterM < iterationM; ++iterM) {
        for (unsigned iterN = 0; iterN < iterationN; ++iterN) {
            for (unsigned i = 0; i < kCount; i++) {
                *reinterpret_cast<float4*>(c + OFFSET(m + i + iterM * intervalM, n + iterN * intervalN, N)) = r_c[iterM][iterN][i]; 
            }
        }
    }
}

int main(void) {
    const int BM = 8, BN = 8;
    const int TEST_M = 1024, TEST_N = 1024, TEST_K = 1024;
    dim3 blockDim_T(BN, BM);
    dim3 gridDim_T(TEST_N / BN / 8, TEST_M / BM / 8);

    void (*gpuSgemm) (float*, float*, float*, const int, const int, const int) = gemm_use_smem<BN, BM>;
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
        dim3 blockDim(BN, BM);
        dim3 gridDim((N / 8 + BN - 1) / BN, (M / 8 + BM - 1) / BM);
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
