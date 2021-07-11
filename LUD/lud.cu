#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdio.h>

#include "../common.h"

void swap(float *A, int i, int j, const int N) {
    for (int k = 0; k < N; k++) {
        float tmp = A[i*N+j];
        A[i*N+k] = A[j*N+k];
        A[j*N+k] = tmp;
    }
}

void luDecomposeHost(float *A, const int N) {
    float sum;
    for (int i=0; i<N; i++) {

        // Find pivot and exchange if necessary.
        int p = i;
        for (int j=i+1; j<N; j++) {
            if (abs(A[j*N+i]) > abs(A[p*N+i]))
                p = j;
        }

        if (p != i) {
            printf("Need Change for this iter    %d -> %d\n", i, p);
            swap(A, p, i, N);
        } else {
            printf("No Need Change for this iter %d\n", i);
        }

        for (int j = i; j < N; j++) {
            sum = A[i*N+j];
            for (int k=0; k<i; k++) {
                sum -= A[i*N+k] * A[k*N+j];
            }
            A[i*N+j] = sum;
        }

        for (int j = i+1; j < N; j++) {
            sum = A[j*N+i];
            for (int k=0; k<i; k++) {
                sum -= A[j*N+k] * A[k*N+i];
            }
            A[j*N+i] = sum / A[i*N+i];
        }
    }
}

const int BLOCK_SIZE = 3;
__global__ void LUDonDeviceRightLooking(float *A) {
    __shared__ float shadow[BLOCK_SIZE][BLOCK_SIZE];
    for (int tix = threadIdx.x; tix<BLOCK_SIZE; tix += blockDim.x) {
        shadow[tix][threadIdx.y] = A[tix * BLOCK_SIZE + threadIdx.y];
        printf("(%d, %d), copy (%d) ->(%d, %d) %f\n", threadIdx.x, threadIdx.y, tix*BLOCK_SIZE + threadIdx.y, tix, threadIdx.y, shadow[tix][threadIdx.y]);
    }
    __syncthreads();

    for (int k=0; k<BLOCK_SIZE; k++) {
        if (threadIdx.y > k && threadIdx.x == k) {
            shadow[threadIdx.y][k] = shadow[threadIdx.y][k] / shadow[k][k];
            printf("(%d, %d), updating (%d, %d) %f\n", threadIdx.x, threadIdx.y, threadIdx.y, k, shadow[threadIdx.y][k]);
        }
        __syncthreads();

        for (int tix = threadIdx.x; tix<BLOCK_SIZE; tix += blockDim.x) {
            if (tix > k && threadIdx.y > k)
                shadow[tix][threadIdx.y] -= shadow[tix][k] * shadow[k][threadIdx.y];
            __syncthreads();
        }
    }

    for (int tix = threadIdx.x; tix<BLOCK_SIZE; tix += blockDim.x) {
        A[tix*BLOCK_SIZE + threadIdx.y] = shadow[tix][threadIdx.y];
    }
}

void printM(float *A, const int N) {
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            printf("%f\t\t", A[i*N+j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void findMaxIndex(float *LU, float *output, int n, int i) {
    extern __shared__ float sdata[];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (row < n) {
        sdata[tid] = LU[row*n+i];
        output[row] = row;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid+s] > sdata[tid]) {
                sdata[tid] = sdata[tid+s];
                output[row] = blockIdx.x * blockDim.x + tid + s;
            }
        }
        __syncthreads();
    }
}

__global__ void swapOnDevice(float *LU, int n, int row) {
    if (max_index == row) {
        return;
    }

    if (threadIdx.x < n) {
        float temp = LU[row*n+threadIdx.x];
        LU[row*n+threadIdx.x] = LU[max_index*n+threadIdx.x];
        LU[max_index*n+threadIdx.x] = temp;
    }
}

__global__ void computeU(float *LU, int n, int i) {
    int col = blockIdx.x * blockDim.x + threadIdx.x + i;
    int row = blockIdx.y + i;
    if (col < n && row < n) {
        float sum = LU[row * n + col];
        for (int k = 0; k < i; k++) {
            sum -= LU[i*n+k] * LU[k*n+col];
        }
        LU[row * n + col] = sum;
    }
}

__global__ void computeL(float *LU, int n, int i) {
    int row = blockIdx.x * blockDim.x + threadIdx.x + i; 
    int col = blockIdx.y + i;
    if (col < n && row < n && row > i) {
        float sum = LU[row*n + col];
        for (int k = 0; k < i; k++) {
            sum -= LU[row*n+k] * LU[k*n+i];
        }
        sum = sum / LU[i*n+i];
        LU[row*n+col] = sum;
    }
}

int luDecomposeNaive(float *LU,int n) {
    int block_size = 64;
    float *dLU;

    size_t nbytes = n*n*sizeof(float);
    cudaMalloc((void**)&dLU, nbytes);
    cudaMemcpy(dLU, LU, nbytes, cudaMemcpyHostToDevice);
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(block_size, 1, 1);
    for (int i=0; i<n; i++) {
        int size = n - i;

        int *dOutput;
        cudaMalloc((void**)&dOutput, size * sizeof(int));
        gridDim.x = ceil((float)size / (float)block_size);
        findMaxInex<<<gridDim, blockDim>>>(dLU, dOutput, n, i);
        int *output = (int*)malloc(size * sizeof(int));
        cudaMemcpy(output, dOutput, size*sizeof(int), cudaMemcpyDeviceToHost);

        int max_index = 0;
        for (int i=0; i<size; i += blockDim.x) {
            if (output[i] > output[max_index]) {
                max_index = i;
            }
        }

        gridDim.x = (n + block_size - 1) / block_size;
        swapOnDevice<<<gridDim, blockDim>>>(dLU, n, i);

        gridDim.x = ceil((float)size / (float)block_size);
        computeU<<<gridDim, blockDim>>>(dLU, n, i);
        computeL<<<gridDim, blockDim>>>(dLU, n, i);
    }
    cudaMemcpy(LU, dLU, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(dLU);
    return 0;
}

#define N 1024

int main(int argc, char** argv) {
    size_t nbytes = N * N * sizeof(float);
    float *A1 = (float*)malloc(nbytes);
    float *A2 = (float*)malloc(nbytes);

    srand((unsigned)time(NULL));
    for (int i = 0; i < N*N; i++) {
        float x = (float)(rand() % 101)/ 101.0f;
        A1[i] = x;
    }
    memcpy(A2, A1, nbytes);

    double iStart, iElaps;

    //printM(A1, N);

    iStart = cpuSecond();
    luDecomposeHost(A1, N);
    iElaps = cpuSecond() - iStart;

    //printM(A1, N);
    printf("luDeComposeHost elapse : %f\n", iElaps);

    //printM(A2, N);
    iStart = cpuSecond();
    luDecomposeNaive(A2, N);
    iElaps = cpuSecond() - iStart;
    printf("luDecomposeNaive elapse: %f\n", iElaps);

    //#printM(A2, N);

    /*dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(1, 1);
    size_t nbytes = BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
    float *d_A = NULL;
    cudaMalloc(&d_A, nbytes);
    cudaMemcpy(d_A, A_copy, nbytes, cudaMemcpyHostToDevice);
    LUDonDeviceRightLooking<<<grid, block>>>(d_A);

    float *h_A = (float*)malloc(nbytes);
    cudaMemcpy(h_A, d_A, nbytes, cudaMemcpyDeviceToHost);
    printM(h_A, 3);*/

}