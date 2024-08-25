#include "inference_engine.h"

#include <iostream>
#include <stdexcept>

#include "onnx_helper.h"
#include "operators.h"

std::string op_type_to_string(OpType op_type);
void applyConstantFolding(Graph &graph);

InferenceEngine::InferenceEngine(
    std::unique_ptr<Graph> graph,
    std::unordered_map<std::string, Tensor<float>> weights)
    : graph_(std::move(graph)), weights_(std::move(weights)) {}

void InferenceEngine::applyOptimizations() { applyConstantFolding(); }

Tensor<float> InferenceEngine::infer(const Tensor<float> &input) {
    weights_[graph_->getInputName(0)] = input;

    for (const auto node : graph_->getTopologicallySortedNodes()) {
        auto inputs = ptrPrepareNodeInputs(node);
        Tensor<float> output = evaluateNode(node, inputs);

        if (output.size() != 0) {
            weights_[node->getOutputs()[0]] = std::move(output);
        } else {
            throw std::runtime_error("Got empty output after inference loop.");
        }
    }

    const auto &graph_output = graph_->getOutputName(0);
    auto it = weights_.find(graph_output);
    if (it == weights_.end()) {
        throw std::runtime_error("Output not found: " + graph_output);
    }

    return it->second;
}
Tensor<float> InferenceEngine::evaluateNode(
    const Node *node, const std::vector<Tensor<float> *> &inputs) {
    switch (node->getOpType()) {
        case OpType::Gemm: {
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
            return Operators<float>::gemm(A, B, bias, transA, transB, alpha,
                                          beta);
        }
        case OpType::Flatten: {
            if (inputs.size() != 1) {
                throw std::runtime_error("Flatten operation expects 1 input");
            }
            auto axisOpt = node->getAttribute<int64_t>("axis");
            if (!axisOpt) {
                throw std::runtime_error("Axis missing for flatten operation");
            }
            return Operators<float>::flatten(*inputs[0], axisOpt.value());
        }
        case OpType::Relu: {
            if (inputs.size() != 1) {
                throw std::runtime_error("Relu operation expects 1 input");
            }
            return Operators<float>::relu(*inputs[0]);
        }
        case OpType::Add: {
            if (inputs.size() != 2) {
                throw std::runtime_error("Add operation expects 2 inputs");
            }
            return Operators<float>::add(*inputs[0], *inputs[1]);
        }
        default:
            throw std::runtime_error("Unsupported op_type: " +
                                     op_type_to_string(node->getOpType()));
    }
}

std::vector<Tensor<float> *> InferenceEngine::ptrPrepareNodeInputs(
    const Node *node) {
    std::vector<Tensor<float> *> inputs;
    const auto &input_names = node->getInputs();
    inputs.reserve(input_names.size());

    for (const auto &input_name : input_names) {
        auto it = weights_.find(input_name);
        if (it == weights_.end()) {
            throw std::runtime_error("Input not found: " + input_name);
        }
        inputs.push_back(&it->second);
    }
    return inputs;
}

void InferenceEngine::applyConstantFolding() {
    for (auto node : graph_->getTopologicallySortedNodes()) {
        std::vector<Tensor<float> *> inputs;
        try {
            inputs = ptrPrepareNodeInputs(node);
            std::cout << "Found constant node, applying constant folding."
                      << std::endl;
        } catch (const std::exception &e) {
            // not a constant node, skip.
            std::cout << "Skipping node, not constant." << std::endl;
            continue;
        }
        Tensor<float> res = evaluateNode(node, inputs);

        // Create constant node with same outputs.
        auto constantNode =
            std::make_unique<Node>(node->getName(), OpType::Constant);
        constantNode->addOutput(constantNode->getOutputs()[0]);
        weights_[constantNode->getOutputs()[0]] = res;
        graph_->replaceNode(node, std::move(constantNode));
    }
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
