#include "gemm_cpu.h"

#include <assert.h>

// gemm_cpu returns out = A * B + bias
// A is (n, m)
// B is (m, k)
// bias is assumed to be (n, 1) and broadcasted
// out is (n, k)
void gemm_cpu(const float* A, const float* B, const float* bias, float* out,
              const int n, const int m, const int k, const bool transA,
              const bool transB, const float alpha, const float beta) {
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < k; ++c) {
            float res = 0;
            for (int i = 0; i < m; ++i) {
                float aVal = transA ? A[i * n + r] : A[r * m + i];
                float bVal = transB ? B[c * m + i] : B[i * k + c];
                res += aVal * bVal;
            }
            out[r * k + c] = res * alpha;
        }
    }

    // Apply bias term
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < k; ++c) {
            out[r * k + c] += bias[r] * beta;
        }
    }
}
