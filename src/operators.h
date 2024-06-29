#ifndef OPERATORS_H
#define OPERATORS_H

#include "tensor.h"

template <typename T>
Tensor<T> flatten(Tensor<T>& tensor, uint64_t axis);

// Tensor gemm(std::vector<const Tensor>& inputs);
// Tensor relu(std::vector<const Tensor>& inputs);


#endif // OPERATORS_H