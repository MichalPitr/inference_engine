#include "inference_engine.h"

#include <iostream>
#include <stdexcept>

#include "onnx_helper.h"
#include "operators.h"
#include "optypes.h"

void applyConstantFolding(Graph &graph);

InferenceEngine::InferenceEngine(DeviceType device) {
    device_ = device;
    registerCpuOperators();
    registerCudaOperators();
}

Tensor<float> InferenceEngine::evaluateNode(
    const Node *node, const std::vector<Tensor<float> *> &inputs) {
    return registry_.executeOperator(node, inputs, device_);
}

void InferenceEngine::registerCpuOperators() {
    registry_.registerCpuOperator(
        OpType::Gemm,
        [](const Node *node, const std::vector<Tensor<float> *> &inputs) {
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
            return CpuOperators<float>::gemm(A, B, bias, transA, transB, alpha,
                                             beta);
        });

    registry_.registerCpuOperator(
        OpType::Flatten,
        [](const Node *node, const std::vector<Tensor<float> *> &inputs) {
            if (inputs.size() != 1) {
                throw std::runtime_error("Flatten operation expects 1 input");
            }
            auto axisOpt = node->getAttribute<int64_t>("axis");
            if (!axisOpt) {
                throw std::runtime_error("Axis missing for flatten operation");
            }
            return CpuOperators<float>::flatten(*inputs[0], axisOpt.value());
        });

    registry_.registerCpuOperator(
        OpType::Relu, []([[maybe_unused]] const Node *node,
                         const std::vector<Tensor<float> *> &inputs) {
            if (inputs.size() != 1) {
                throw std::runtime_error("Relu operation expects 1 input");
            }
            return CpuOperators<float>::relu(*inputs[0]);
        });

    registry_.registerCpuOperator(
        OpType::Add, []([[maybe_unused]] const Node *node,
                        const std::vector<Tensor<float> *> &inputs) {
            if (inputs.size() != 2) {
                throw std::runtime_error("Add operation expects 2 inputs");
            }
            return CpuOperators<float>::add(*inputs[0], *inputs[1]);
        });
}

void InferenceEngine::registerCudaOperators() {
    registry_.registerCudaOperator(
        OpType::Gemm,
        [](const Node *node, const std::vector<Tensor<float> *> &inputs) {
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
            return CudaOperators<float>::gemm(A, B, bias, transA, transB, alpha,
                                              beta);
        });

    registry_.registerCudaOperator(
        OpType::Flatten,
        [](const Node *node, const std::vector<Tensor<float> *> &inputs) {
            if (inputs.size() != 1) {
                throw std::runtime_error("Flatten operation expects 1 input");
            }
            auto axisOpt = node->getAttribute<int64_t>("axis");
            if (!axisOpt) {
                throw std::runtime_error("Axis missing for flatten operation");
            }
            return CudaOperators<float>::flatten(*inputs[0], axisOpt.value());
        });

    registry_.registerCudaOperator(
        OpType::Relu, []([[maybe_unused]] const Node *node,
                         const std::vector<Tensor<float> *> &inputs) {
            if (inputs.size() != 1) {
                throw std::runtime_error("Relu operation expects 1 input");
            }
            return CudaOperators<float>::relu(*inputs[0]);
        });

    registry_.registerCudaOperator(
        OpType::Add, []([[maybe_unused]] const Node *node,
                        const std::vector<Tensor<float> *> &inputs) {
            if (inputs.size() != 2) {
                throw std::runtime_error("Add operation expects 2 inputs");
            }
            return CudaOperators<float>::add(*inputs[0], *inputs[1]);
        });
}
