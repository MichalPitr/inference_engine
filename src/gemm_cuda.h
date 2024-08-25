#ifndef GEMM_CUDA_H
#define GEMM_CUDA_H

void gemm_cuda(const float* A, const float* B, const float* bias, float* out,
               int n, int m, int k, bool transA, bool transB, float alpha,
               float beta);

#endif  // GEMM_CUDA_H
