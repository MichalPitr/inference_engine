#include "operator_registry.h"

#include <stdexcept>

#include "optypes.h"

template <typename T>
void OperatorRegistry<T>::registerOperator(OpType opType,
                                           OperatorFunction func) {
    operatorMap[opType] = std::move(func);
}

template <typename T>
Tensor<T> OperatorRegistry<T>::executeOperator(
    const Node* node, const std::vector<Tensor<T>*>& inputs) {
    auto it = operatorMap.find(node->getOpType());
    if (it == operatorMap.end()) {
        throw std::runtime_error(
            "Unsupported op_type: " +
            std::to_string(static_cast<int>(node->getOpType())));
    }
    return it->second(node, inputs);
}

template class OperatorRegistry<float>;