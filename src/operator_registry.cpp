#include "operator_registry.h"

#include <stdexcept>

#include "optypes.h"

std::string op_type_to_string(OpType op_type);

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

std::string op_type_to_string(OpType op_type) {
    switch (op_type) {
        case OpType::Input:
            return "Input";
        case OpType::Output:
            return "Output";
        case OpType::Add:
            return "Add";
        case OpType::Gemm:
            return "Gemm";
        case OpType::Flatten:
            return "Flatten";
        case OpType::Relu:
            return "Relu";
        case OpType::Conv:
            return "Conv";
        case OpType::MaxPool:
            return "MaxPool";
        default:
            throw std::runtime_error("Unknown op_type");
    }
}

template class OperatorRegistry<float>;