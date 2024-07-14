#ifndef GEMM_H
#define GEMM_H

// Function declaration
void gemm(const float* A, const float* B, const float* bias, float* out, 
    const int m, const int n, const int k, 
    const bool transA, const bool transB, const float alpha, const float beta);

#endif // GEMM_H