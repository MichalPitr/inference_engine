#ifndef OPERATOR_REGISTRY_H
#define OPERATOR_REGISTRY_H

#include <functional>

#include "node.h"
#include "optypes.h"
#include "tensor.h"

template <typename T>
class OperatorRegistry {
   public:
    using OperatorFunction =
        std::function<Tensor<T>(const Node*, const std::vector<Tensor<T>*>&)>;

    void registerOperator(OpType opType, OperatorFunction func);
    Tensor<T> executeOperator(const Node* node,
                              const std::vector<Tensor<T>*>& inputs);

   private:
    std::unordered_map<OpType, OperatorFunction> operatorMap;
};
#endif