#ifndef EXECUTION_PROVIDER_H
#define EXECUTION_PROVIDER_H

#include "node.h"
#include "tensor.h"

class ExecutionProvider {
   public:
    ExecutionProvider() = default;
    virtual ~ExecutionProvider() = default;
    virtual Tensor<float> evaluateNode(
        const Node* node, const std::vector<Tensor<float>*>& inputs) = 0;
};

#endif  // EXECUTION_PROVIDER_H