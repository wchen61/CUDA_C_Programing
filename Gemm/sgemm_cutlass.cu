#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm.h"
#include "utils.cuh"

float testCutlassError(const int M, const int N, const int K) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float*)malloc(size_a);
    h_b = (float*)malloc(size_b);
    h_c = (float*)malloc(size_c);
    checkCudaErrors(cudaMalloc(&d_a, size_a));
    checkCudaErrors(cudaMalloc(&d_b, size_b));
    checkCudaErrors(cudaMalloc(&d_c, size_c));
    h_d_c = (float*)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        h_b[i] = (float)rand() / RAND_MAX;
    }
    checkCudaErrors(cudaMemset(d_c, 0, size_c));
    cpuSgemm(h_a, h_b, h_c, M, N, K);
    /*printData("A", h_a, K, M);
    printData("B", h_b, N, K);
    printData("C", h_c, N, M);*/
 
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    using RowMajor = cutlass::layout::RowMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float, RowMajor>;
    CutlassGemm gemm_operator;
    float alpha = 1.0;
    float beta = 0.0;
    CutlassGemm::Arguments args({M, N, K},
                                {d_a, K},
                                {d_b, N},
                                {d_c, N},
                                {d_c, N},
                                {alpha, beta}); 
    cutlass::Status status = gemm_operator(args);

    if (status != cutlass::Status::kSuccess) {
        printf("Cutlass Error");
        return -1;
    }

    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    //printData("d_C", h_d_c, N, M);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) {
            max_error = -NAN;
        } else {
            max_error = max(max_error, this_error);
        }
    }


    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);
    return max_error;
}

float testCutlassPerformance(const int M, const int N, const int K, const int repeat) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    checkCudaErrors(cudaMalloc(&d_a, size_a));
    checkCudaErrors(cudaMalloc(&d_b, size_b));
    checkCudaErrors(cudaMalloc(&d_c, size_c));

    using RowMajor = cutlass::layout::RowMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float, RowMajor>;
    CutlassGemm gemm_operator;
    float alpha = 1.0;
    float beta = 0.0;
    CutlassGemm::Arguments args({M, N, K},
                                {d_a, K},
                                {d_b, N},
                                {d_c, N},
                                {d_c, N},
                                {alpha, beta}); 
    cutlass::Status status = gemm_operator(args);
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    for (int i = 0; i < repeat; i++) {
        cutlass::Status status = gemm_operator(args);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return sec;
}

int main(void) {
    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    const int TEST_M = 1024, TEST_N = 1024, TEST_K = 1024;
    const int BLOCK_SIZE = (BN / TN) * (BM / TM);
    dim3 blockDim_T(BN / TN, BM / TM);
    dim3 gridDim_T(TEST_N / BN, TEST_M / BM);

    float max_error = testCutlassError(TEST_M, TEST_N, TEST_K);
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
        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j++) {
            double this_sec = testCutlassPerformance(M, N, K, inner_repeat);
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