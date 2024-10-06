#include <cuda_runtime.h>

#include <iostream>

#define BLOCK_SIZE 16
#define TILE_SIZE 16

/*
Tensors are of shape:
A: (n, m)
B: (m, k)
C: (n, k)
*/
__global__ void gemm_kernel_naive(const float *A, const float *B,
                                  const float *bias, float *out, int n, int m,
                                  int k, bool transA, bool transB, float alpha,
                                  float beta) {
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;

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

__global__ void gemm_kernel_tiled(const float *A, const float *B,
                                  const float *bias, float *out, int n, int m,
                                  int k, bool transA, bool transB, float alpha,
                                  float beta) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    uint tx = threadIdx.x;
    uint ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float res = 0.0f;

    for (int t = 0; t < (m + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Collaboratively load tile
        if (row < n && t * TILE_SIZE + tx < m) {
            As[ty][tx] = transA ? A[(t * TILE_SIZE + tx) * n + row]
                                : A[row * m + (t * TILE_SIZE + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (t * TILE_SIZE + ty < m && col < k) {
            Bs[ty][tx] = transB ? B[col * m + (t * TILE_SIZE + ty)]
                                : B[(t * TILE_SIZE + ty) * k + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Matmul over tile
        for (int i = 0; i < TILE_SIZE; ++i) {
            res += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < n && col < k) {
        out[row * k + col] = res * alpha + bias[col] * beta;
    }
}

void gemm_cuda_tiled(const float *A, const float *B, const float *bias,
                     float *out, int n, int m, int k, bool transA, bool transB,
                     float alpha, float beta) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((k + TILE_SIZE - 1) / TILE_SIZE,
                 (n + TILE_SIZE - 1) / TILE_SIZE);

    gemm_kernel_tiled<<<gridDim, blockDim>>>(A, B, bias, out, n, m, k, transA,
                                             transB, alpha, beta);
}

void gemm_cuda_naive(const float *A, const float *B, const float *bias,
                     float *out, int n, int m, int k, bool transA, bool transB,
                     float alpha, float beta) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim((k + blockDim.x - 1) / blockDim.x,
                 (n + blockDim.y - 1) / blockDim.y);

    gemm_kernel_naive<<<gridDim, blockDim>>>(A, B, bias, out, n, m, k, transA,
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

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x,
                  (n + blockSize.y - 1) / blockSize.y);

    gemm_kernel_naive<<<gridSize, blockSize>>>(d_A, d_B, d_bias, d_out, n, m, k,
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