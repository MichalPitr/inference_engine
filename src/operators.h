#ifndef OPERATORS_H
#define OPERATORS_H

#include "tensor.h"

template <typename T>
class CpuOperators {
   public:
    static Tensor<T> gemm(const Tensor<T>& A, const Tensor<T>& B,
                          const Tensor<T>& bias, bool transA, bool transB,
                          float alpha, float beta);
    static Tensor<T> flatten(Tensor<T>& tensor, uint64_t axis);
    static Tensor<T> relu(const Tensor<T>& tensor);
    static Tensor<T> add(const Tensor<T>& A, const Tensor<T>& B);
};

template <typename T>
class CudaOperators {
   public:
    static Tensor<T> gemm(const Tensor<T>& A, const Tensor<T>& B,
                          const Tensor<T>& bias, bool transA, bool transB,
                          float alpha, float beta);
    static Tensor<T> flatten(Tensor<T>& tensor, uint64_t axis);
    static Tensor<T> relu(const Tensor<T>& tensor);
    static Tensor<T> add(const Tensor<T>& A, const Tensor<T>& B);
};

#endif  // OPERATORS_H