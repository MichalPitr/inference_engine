#include "gtest/gtest.h"
#include "../gemm.h" 

TEST(GemmTest, NoBias) {
    const int n = 2; 
    const int m = 2;
    const int k = 1; // Must be 1 for this gemm implementation

    float A[n * m] = {1.0, 1.0, 1.0, 1.0};
    float B[m * k] = {2.0, 3.0};
    float C[n] = {}; // 0 Bias vector

    float expected[n] = {5.0, 5.0};
    float out[n];

    gemm(A, B, C, out, n, m, k);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(out[i], expected[i], 1e-5);  // Allow small tolerance for floating-point errors
    }
}

TEST(GemmTest, MatrixVectorMultiplication) {
    const int n = 2; 
    const int m = 2;
    const int k = 1; // Must be 1 for this gemm implementation

    float A[n * m] = {1.0, 2.0, 3.0, 4.0};
    float B[m * k] = {5.0, 6.0};
    float C[n]     = {7.0, 8.0}; // Bias vector

    float expected[n] = {24.0, 47.0};
    float out[n];

    gemm(A, B, C, out, n, m, k);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(out[i], expected[i], 1e-5);  // Allow small tolerance for floating-point errors
    }
}

TEST(GemmTest, DifferentDimensions) {
    const int n = 2; 
    const int m = 3;
    const int k = 1; 

    float A[n * m] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    float B[m * k] = {2.0, 3.0, 4.0};
    float C[n]     = {1.0, 1.0};

    float expected[n] = {10.0, 10.0};
    float out[n];

    gemm(A, B, C, out, n, m, k);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(out[i], expected[i], 1e-5);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}