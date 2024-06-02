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

    float A[n * m] = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0};
    float B[m * k] = {2.0, 3.0, 4.0};
    float C[n]     = {1.0, 1.0};

    float expected[n] = {10.0, 19.0};
    float out[n];

    gemm(A, B, C, out, n, m, k);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(out[i], expected[i], 1e-5);
    }
}

// Generated with python.
TEST(GemmTest, LargeMatrix) {
    const int n = 10; 
    const int m = 5;
    const int k = 1; 

    float A[n * m] = {0.37406192933141147, 0.06745353143398714, 0.1360688163646061, 0.7562505073146693, 0.5428856927950624, 0.31954021763742935, 0.8946841711756393, 0.7743437586367583, 0.14352502492052077, 0.41265414487951024, 0.8296688316398949, 0.27251306465433534, 0.14776552565735301, 0.3950728124771392, 0.7687820220977069, 0.16392780073148927, 0.7402754740346736, 0.7243936043094837, 0.8633784327013307, 0.3487963299151927, 0.6669684460103615, 0.3531235616889036, 0.9760224780679175, 0.7263405530192332, 0.7516497260607778, 0.28503760873098205, 0.46137578945845437, 0.20012238575099484, 0.10529354612903685, 0.9239620151968889, 0.6608644427436974, 0.5586669905827026, 0.9818996653233024, 0.8885926029218781, 0.984107985981659, 0.0027823158763550238, 0.4801631350272688, 0.34023762153957926, 0.9783240486545588, 0.4044755719976537, 0.23219208762135335, 0.7583056863535441, 0.5877358671639532, 0.9278162173103481, 0.418900146932737, 0.2351060992425743, 0.5169877034859311, 0.4746289284042273, 0.4689403087502675, 0.08573114728236486};
    float B[m * k] = {0.6388001442740947, 0.7295867686529213, 0.022577391563295524, 0.3353350568167005, 0.1839017387537819};
    float C[n]     = {0.9533634522173151, 0.6163495493570798, 0.183166574752734, 0.4766469611276871, 0.2677701486587737, 0.2026184257272713, 0.8402237268446139, 0.6291182992377615, 0.6915170468341604, 0.4516003466136026};

    float expected[n] = {1.5980344793090224, 1.6147210692751712, 1.1891793450983736, 1.4914794802764628, 1.3552971503375555, 0.9310592425482995, 2.1711042265782834, 1.391348102552378, 1.794526493404922, 1.1627076600419188};
    float out[n];

    gemm(A, B, C, out, n, m, k);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(out[i], expected[i], 1e-5);
    }
}

TEST(GemmTest, AllZeroes) {
    const int n = 4; 
    const int m = 3;
    const int k = 1; 

    float A[n * m] = {};
    float B[m * k] = {};
    float C[n]     = {};

    float expected[n] = {};
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