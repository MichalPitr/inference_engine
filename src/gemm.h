#ifndef GEMM_H
#define GEMM_H

void gemm(const float* A, const float* B, const float* bias, float* out, 
    const int m, const int n, const int k, 
    const bool transA, const bool transB, const float alpha, const float beta);

#endif // GEMM_H