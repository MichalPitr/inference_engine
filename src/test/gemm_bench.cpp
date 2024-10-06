#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "../kernels.h"

// Helper function to initialize a matrix with random values
void initializeMatrix(std::vector<float>& matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Benchmark function
static void BM_GEMM(benchmark::State& state) {
    const int n = state.range(0);
    const int m = state.range(1);
    const int k = state.range(2);
    const bool is_tiled = state.range(3);

    // Allocate and initialize host matrices
    std::vector<float> h_A(n * m), h_B(m * k), h_bias(k), h_out(n * k);
    initializeMatrix(h_A, n * m);
    initializeMatrix(h_B, m * k);
    initializeMatrix(h_bias, k);

    // Allocate device memory
    float *d_A, *d_B, *d_bias, *d_out;
    cudaMalloc(&d_A, n * m * sizeof(float));
    cudaMalloc(&d_B, m * k * sizeof(float));
    cudaMalloc(&d_bias, k * sizeof(float));
    cudaMalloc(&d_out, n * k * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), k * sizeof(float),
               cudaMemcpyHostToDevice);

    // Benchmark loop
    for (auto _ : state) {
        // Create a CUDA event to measure GPU time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Record the start event
        cudaEventRecord(start, nullptr);

        if (is_tiled) {
            gemm_cuda_tiled(d_A, d_B, d_bias, d_out, n, m, k, false, false,
                            1.0f, 1.0f);
        } else {
            gemm_cuda_naive(d_A, d_B, d_bias, d_out, n, m, k, false, false,
                            1.0f, 1.0f);
        }

        // Record the stop event
        cudaEventRecord(stop, nullptr);
        cudaEventSynchronize(stop);

        // Calculate the elapsed time in milliseconds
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Destroy the events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Report the time
        state.SetIterationTime(milliseconds / 1000.0);
    }

    // Calculate and report throughput
    state.SetBytesProcessed(int64_t(state.iterations()) * n * m * k *
                            sizeof(float));
    state.SetItemsProcessed(int64_t(state.iterations()) * n * m * k);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_bias);
    cudaFree(d_out);
}

// Define the benchmark
BENCHMARK(BM_GEMM)
    ->Args({4092, 4092, 4092, 0})  // n, m, k, (0 for naive)
    ->Args({4092, 4092, 4092, 1})  // n, m, k, (1 for tiled)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK_MAIN();