#ifndef OPERATORS_H
#define OPERATORS_H

#include "tensor.h"

template <typename T>
Tensor<T> flatten(Tensor<T> &tensor, uint64_t axis);

template <typename T>
Tensor<T> relu(Tensor<T> &tensor);

template <typename T>
Tensor<T> add(Tensor<T> &A, Tensor<T> &B);

template <typename T>
Tensor<T> gemm(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &bias, bool transA, bool transB, float alpha, float beta);

#endif // OPERATORS_H