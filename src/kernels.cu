#include <cuda_runtime.h>

#include <iostream>

__global__ void gemm_kernel(const float *A, const float *B, const float *bias,
                            float *out, int n, int m, int k, bool transA,
                            bool transB, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < k) {
        float res = 0.0f;

        for (int i = 0; i < m; ++i) {
            float aVal = transA ? A[i * n + row] : A[row * m + i];
            float bVal = transB ? B[col * m + i] : B[i * k + col];
            res += aVal * bVal;
        }
        out[row * k + col] = res * alpha + bias[row] * beta;
    }
}

void gemm_cuda(const float *A, const float *B, const float *bias, float *out,
               int n, int m, int k, bool transA, bool transB, float alpha,
               float beta) {
    dim3 blockSize(16, 16);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x,
                  (n + blockSize.y - 1) / blockSize.y);

    gemm_kernel<<<gridSize, blockSize>>>(A, B, bias, out, n, m, k, transA,
                                         transB, alpha, beta);
}

__global__ void relu_kernel(float *X, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        X[idx] = X[idx] < 0 ? 0 : X[idx];
    }
}

void relu_cuda(float *X, int n) { relu_kernel<<<ceil(n / 32.0), 32>>>(X, n); }

__global__ void add_kernel(float *A, const float *B, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx] += B[idx];
    }
}

void add_cuda(float *A, const float *B, int n) {
    add_kernel<<<ceil(n / 32.0), 32>>>(A, B, n);
}