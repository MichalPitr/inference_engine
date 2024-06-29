#ifndef GEMM_H
#define GEMM_H

// Function declaration
void gemm(const float* A, const float* B, const float* bias, float* out, int m, int n, int k);

#endif // GEMM_H