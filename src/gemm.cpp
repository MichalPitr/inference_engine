#include <assert.h>

#include "gemm.h"

// gemm returns out = A * B + bias
// A is (n, m)
// B is (m, k)
// bias is (n, 1)
// out is (n, k)
void gemm(const float *A, const float *B, const float *bias, float *out, const int n, const int m, const int k)
{
    for (int r = 0; r < n; ++r)
    {
        for (int c = 0; c < k; ++c)
        {
            float res = 0;
            for (int i = 0; i < m; ++i)
            {
                res += A[r * m + i] * B[i * k + c];
            }
            out[r * k + c] = res;
        }
    }

    // assert(k == 1);
    for (int r = 0; r < n; ++r)
    {
        for (int c = 0; c < k; ++c) {
            out[r*k + c] += bias[r];
        }
    }
}
