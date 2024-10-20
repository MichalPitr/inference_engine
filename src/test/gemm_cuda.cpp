#include <cuda_runtime.h>

#include <cmath>
#include <random>

#include "../kernels.h"

// Helper function to initialize a matrix with random values
void initializeRandomMatrix(float* matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
}

// Reference CPU implementation of matrix multiplication
void gemm_cpu_reference(const float* A, const float* B, const float* bias,
                        float* C, int n, int m, int k, bool transA, bool transB,
                        float alpha, float beta) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < m; ++l) {
                int a_idx = transA ? l * n + i : i * m + l;
                int b_idx = transB ? j * m + l : l * k + j;
                sum += A[a_idx] * B[b_idx];
            }
            C[i * k + j] = alpha * sum + beta * (bias ? bias[j] : 0.0f);
        }
    }
}

void runGemmTest(int n, int m, int k, bool transA, bool transB, float alpha,
                 float beta) {
    size_t sizeA = n * m;
    size_t sizeB = m * k;
    size_t sizeC = n * k;

    // Allocate host memory
    std::vector<float> h_A(sizeA);
    std::vector<float> h_B(sizeB);
    std::vector<float> h_bias(k);
    std::vector<float> h_C_cuda(sizeC);
    std::vector<float> h_C_cpu(sizeC);

    // Initialize matrices
    initializeRandomMatrix(h_A.data(), n, m);
    initializeRandomMatrix(h_B.data(), m, k);
    initializeRandomMatrix(h_bias.data(), 1, k);

    // Allocate device memory
    float *d_A, *d_B, *d_bias, *d_C;
    cudaMalloc(&d_A, sizeA * sizeof(float));
    cudaMalloc(&d_B, sizeB * sizeof(float));
    cudaMalloc(&d_bias, k * sizeof(float));
    cudaMalloc(&d_C, sizeC * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), k * sizeof(float),
               cudaMemcpyHostToDevice);

    // Run CUDA kernel
    gemm_tiled_1D_blocktiling(d_A, d_B, d_bias, d_C, n, m, k, transA, transB,
                              alpha, beta);

    // Copy result back to host
    cudaMemcpy(h_C_cuda.data(), d_C, sizeC * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Run CPU reference implementation
    gemm_cpu_reference(h_A.data(), h_B.data(), h_bias.data(), h_C_cpu.data(), n,
                       m, k, transA, transB, alpha, beta);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_bias);
    cudaFree(d_C);
}

int main() { runGemmTest(64, 64, 64, false, false, 1.0f, 1.0f); }