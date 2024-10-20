#ifndef KERNELS_H
#define KERNELS_H

void gemm_cuda_tiled(const float* A, const float* B, const float* bias,
                     float* out, int n, int m, int k, bool transA, bool transB,
                     float alpha, float beta);

void gemm_cuda_tiled_1D(const float* A, const float* B, const float* bias,
                        float* out, int n, int m, int k, bool transA,
                        bool transB, float alpha, float beta);

void gemm_tiled_1D_blocktiling(const float* A, const float* B,
                               const float* bias, float* out, int n, int m,
                               int k, bool transA, bool transB, float alpha,
                               float beta);

void gemm_cuda_naive(const float* A, const float* B, const float* bias,
                     float* out, int n, int m, int k, bool transA, bool transB,
                     float alpha, float beta);

void gemm_cuda_unoptimized(const float* A, const float* B, const float* bias,
                           float* out, int n, int m, int k, bool transA,
                           bool transB, float alpha, float beta);

void relu_cuda(const float* in, float* out, int n);

void relu_cuda_unoptimized(const float* in, float* out, int n);

void add_cuda(const float* A, const float* B, float* out, int n);

void add_cuda_unoptimized(const float* A, const float* B, float* out, int n);

#endif  // KERNELS_H
