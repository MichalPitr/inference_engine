#ifndef KERNELS_H
#define KERNELS_H

void gemm_cuda(const float* A, const float* B, const float* bias, float* out,
               int n, int m, int k, bool transA, bool transB, float alpha,
               float beta);

void relu_cuda(float* X, int n);

void add_cuda(float* A, const float* B, int n);

#endif  // KERNELS_H
