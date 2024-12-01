#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "../kernels.h"

#define TEST_GEMM(test_name, n, m, k)                   \
    TEST_F(GemmCudaTest, test_name##_NoTranspose) {     \
        runGemmTest(n, m, k, false, false, 1.0f, 1.0f); \
    }                                                   \
    TEST_F(GemmCudaTest, test_name##_TransposeB) {      \
        runGemmTest(n, m, k, false, true, 1.0f, 1.0f);  \
    }

// Helper function to initialize a matrix with random values
void initializeRandomMatrix(float* matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
}

// Helper function to compare matrices
void compareMatrices(const float* A, const float* B, int size,
                     float tolerance = 1e-5) {
    for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(
            A[i], B[i],
            tolerance);  // Allow small tolerance for floating-point errors
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

class GemmCudaTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // CUDA initialization if needed
    }

    void TearDown() override {
        // CUDA cleanup if needed
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
        cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias, h_bias.data(), k * sizeof(float),
                   cudaMemcpyHostToDevice);

        // Run CUDA kernel
        gemm_tiled_1D_blocktiling(d_A, d_B, d_bias, d_C, n, m, k, transA,
                                  transB, alpha, beta);

        // Copy result back to host
        cudaMemcpy(h_C_cuda.data(), d_C, sizeC * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // Run CPU reference implementation
        gemm_cpu_reference(h_A.data(), h_B.data(), h_bias.data(),
                           h_C_cpu.data(), n, m, k, transA, transB, alpha,
                           beta);

        // Compare results
        compareMatrices(h_C_cuda.data(), h_C_cpu.data(), sizeC);

        // std::cout << "cuda:\n";
        // for (int y = 0; y < k; ++y) {
        //     for (int x = 0; x < n; ++x) {
        //         std::cout << h_C_cuda[y * n + x] << " ";
        //     }
        //     std::cout << "\n";
        // }

        // std::cout << "cpu:\n";
        // for (int y = 0; y < k; ++y) {
        //     for (int x = 0; x < n; ++x) {
        //         std::cout << h_C_cpu[y * n + x] << " ";
        //     }
        //     std::cout << "\n";
        // }

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_bias);
        cudaFree(d_C);
    }
};
TEST_GEMM(OneSquare, 1, 1, 1)
TEST_GEMM(TwoSquare, 2, 2, 2)
TEST_GEMM(FourSquare, 4, 4, 4)
TEST_GEMM(FiveSquare, 5, 5, 5)
TEST_GEMM(SixteenSquare, 16, 16, 16)
TEST_GEMM(ThirtyTwoSquare, 32, 32, 32)
TEST_GEMM(SixtyFourSquare, 64, 64, 64)
TEST_GEMM(LargeSquare, 65, 65, 65)
TEST_GEMM(LargerSquare, 128, 128, 128)
TEST_GEMM(BigSquare, 1028, 1028, 1028)
TEST_GEMM(SmallNonSquare, 2, 4, 8)
TEST_GEMM(MediumNonSquare, 31, 15, 43)
TEST_GEMM(NonSquare, 6, 4, 2)
TEST_GEMM(BigNonSquare, 333, 1027, 717)

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}