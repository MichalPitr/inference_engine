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

    void registerCpuOperator(OpType opType, OperatorFunction func);
    void registerCudaOperator(OpType opType, OperatorFunction func);
    Tensor<T> executeOperator(const Node* node,
                              const std::vector<Tensor<T>*>& inputs,
                              const DeviceType device);

   private:
    std::unordered_map<OpType, OperatorFunction> cpuOperatorMap;
    std::unordered_map<OpType, OperatorFunction> cudaOperatorMap;
};
#endif