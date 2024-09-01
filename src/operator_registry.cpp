#include "operator_registry.h"

#include <stdexcept>

#include "optypes.h"

template <typename T>
void OperatorRegistry<T>::registerCpuOperator(OpType opType,
                                              OperatorFunction func) {
    cpuOperatorMap[opType] = std::move(func);
}

template <typename T>
void OperatorRegistry<T>::registerCudaOperator(OpType opType,
                                               OperatorFunction func) {
    cudaOperatorMap[opType] = std::move(func);
}

template <typename T>
Tensor<T> OperatorRegistry<T>::executeOperator(
    const Node* node, const std::vector<Tensor<T>*>& inputs,
    const DeviceType device) {
    auto& operatorMap =
        (device == DeviceType::CUDA) ? cudaOperatorMap : cpuOperatorMap;
    auto it = operatorMap.find(node->getOpType());

    if (it == operatorMap.end()) {
        throw std::runtime_error(
            "Unsupported op_type: " +
            std::to_string(static_cast<int>(node->getOpType())) +
            " for device: " + std::to_string(static_cast<int>(device)));
    }

    return it->second(node, inputs);
}

template class OperatorRegistry<float>;