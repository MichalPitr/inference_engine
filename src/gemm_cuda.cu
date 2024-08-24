#include <iostream>

#include <cuda_runtime.h>

__global__ void gemm_kernel(const float *A, const float *B, const float *bias, float *out, int n, int m, int k, bool transA, bool transB, float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < k)
    {
        float res = 0.0f;

        for (int i = 0; i < m; ++i)
        {
            float aVal = transA ? A[i * n + row] : A[row * m + i];
            float bVal = transB ? B[col * m + i] : B[i * k + col];
            res += aVal * bVal;
        }
        out[row * k + col] = res * alpha + bias[row] * beta;
    }
}


void gemm_cuda(const float *A, const float *B, const float *bias, float *out, int n, int m, int k, bool transA, bool transB, float alpha, float beta)
{
    float *d_A, *d_B, *d_bias, *d_out;

    cudaMalloc((void **)&d_A, n * m * sizeof(float));
    cudaMalloc((void **)&d_B, m * k * sizeof(float));
    cudaMalloc((void **)&d_bias, n * sizeof(float));
    cudaMalloc((void **)&d_out, n * k * sizeof(float));

    cudaMemcpy(d_A, A, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    gemm_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_bias, d_out, n, m, k, transA, transB, alpha, beta);

    cudaMemcpy(out, d_out, n * k * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_bias);
    cudaFree(d_out);
}