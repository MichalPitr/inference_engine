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
        out[row * k + col] = res * alpha + bias[col] * beta;
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

void gemm_cuda_unoptimized(const float *A, const float *B, const float *bias,
                           float *out, int n, int m, int k, bool transA,
                           bool transB, float alpha, float beta) {
    float *d_A, *d_B, *d_bias, *d_out;

    cudaMalloc((void **)&d_A, n * m * sizeof(float));
    cudaMalloc((void **)&d_B, m * k * sizeof(float));
    cudaMalloc((void **)&d_bias, k * sizeof(float));
    cudaMalloc((void **)&d_out, n * k * sizeof(float));

    cudaMemcpy(d_A, A, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x,
                  (n + blockSize.y - 1) / blockSize.y);

    gemm_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_bias, d_out, n, m, k,
                                         transA, transB, alpha, beta);

    cudaMemcpy(out, d_out, n * k * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_bias);
    cudaFree(d_out);
}

__global__ void relu_kernel(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] < 0 ? 0 : in[idx];
    }
}

void relu_cuda(const float *in, float *out, int n) {
    relu_kernel<<<ceil(n / 32.0), 32>>>(in, out, n);
}

void relu_cuda_unoptimized(const float *in, float *out, int n) {
    float *d_in, *d_out;
    cudaMalloc((void **)&d_in, n * sizeof(float));
    cudaMalloc((void **)&d_out, n * sizeof(float));

    cudaMemcpy(d_in, in, n * sizeof(float), cudaMemcpyHostToDevice);

    relu_kernel<<<ceil(n / 32.0), 32>>>(d_in, d_out, n);

    cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

__global__ void add_kernel(const float *A, const float *B, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = A[idx] + B[idx];
    }
}

void add_cuda(const float *A, const float *B, float *out, int n) {
    add_kernel<<<ceil(n / 32.0), 32>>>(A, B, out, n);
}

void add_cuda_unoptimized(const float *A, const float *B, float *out, int n) {
    float *d_A, *d_B, *d_out;
    cudaMalloc((void **)&d_A, n * sizeof(float));
    cudaMalloc((void **)&d_B, n * sizeof(float));
    cudaMalloc((void **)&d_out, n * sizeof(float));

    cudaMemcpy(d_A, A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * sizeof(float), cudaMemcpyHostToDevice);

    add_kernel<<<ceil(n / 32.0), 32>>>(d_A, d_B, d_out, n);

    cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);
}