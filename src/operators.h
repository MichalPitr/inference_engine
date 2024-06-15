#ifndef OPERATORS_H
#define OPERATORS_H

#include <vector>
#include "tensor.h"

Tensor* gemm(std::vector<const Tensor*>& inputs);
Tensor* flatten(std::vector<const Tensor*>& inputs, uint64_t axis);
Tensor* relu(std::vector<const Tensor*>& inputs);


#endif // OPERATORS_H