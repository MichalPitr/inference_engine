#include "cpu_provider.h"

#include <iostream>
#include <stdexcept>

#include "onnx_helper.h"
#include "operators.h"
#include "optypes.h"

Tensor<float> CpuProvider::evaluateNode(
    const Node *node, const std::vector<Tensor<float> *> &inputs) {
    switch (node->getOpType()) {
        case OpType::Gemm:
            return gemm(node, inputs);
        case OpType::Flatten:
            return flatten(node, inputs);
        case OpType::Relu:
            return relu(node, inputs);
        case OpType::Add:
            return add(node, inputs);
        default:
            throw std::runtime_error("Unsupported operation type");
    }
}

Tensor<float> CpuProvider::gemm(const Node *node,
                                const std::vector<Tensor<float> *> &inputs) {
    float alpha = node->getAttribute<float>("alpha").value_or(1.0);
    float beta = node->getAttribute<float>("beta").value_or(1.0);
    int transA = node->getAttribute<int64_t>("transA").value_or(0);
    int transB = node->getAttribute<int64_t>("transB").value_or(0);

    if (inputs.size() != 3) {
        throw std::runtime_error("Gemm operation expects 3 inputs");
    }
    const Tensor<float> &A = *inputs[0];
    const Tensor<float> &B = *inputs[1];
    const Tensor<float> &bias = *inputs[2];
    return CpuOperators<float>::gemm(A, B, bias, transA, transB, alpha, beta);
}

Tensor<float> CpuProvider::flatten(const Node *node,
                                   const std::vector<Tensor<float> *> &inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("Flatten operation expects 1 input");
    }
    auto axisOpt = node->getAttribute<int64_t>("axis");
    if (!axisOpt) {
        throw std::runtime_error("Axis missing for flatten operation");
    }
    return CpuOperators<float>::flatten(*inputs[0], axisOpt.value());
}

Tensor<float> CpuProvider::relu([[maybe_unused]] const Node *node,
                                const std::vector<Tensor<float> *> &inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("Relu operation expects 1 input");
    }
    return CpuOperators<float>::relu(*inputs[0]);
}

Tensor<float> CpuProvider::add([[maybe_unused]] const Node *node,
                               const std::vector<Tensor<float> *> &inputs) {
    if (inputs.size() != 2) {
        throw std::runtime_error("Add operation expects 2 inputs");
    }
    return CpuOperators<float>::add(*inputs[0], *inputs[1]);
}
