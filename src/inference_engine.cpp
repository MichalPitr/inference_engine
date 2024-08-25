#include "inference_engine.h"

#include <iostream>
#include <stdexcept>

#include "onnx_helper.h"
#include "operators.h"
#include "optypes.h"

void applyConstantFolding(Graph &graph);

InferenceEngine::InferenceEngine(
    std::unique_ptr<Graph> graph,
    std::unordered_map<std::string, Tensor<float>> weights)
    : graph_(std::move(graph)), weights_(std::move(weights)) {
    registerOperators();
}

Tensor<float> InferenceEngine::infer(const Tensor<float> &input) {
    weights_[graph_->getInputName(0)] = input;

    for (const auto node : graph_->getTopologicallySortedNodes()) {
        auto inputs = prepareNodeInputs(node);
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
    return registry_.executeOperator(node, inputs);
}

std::vector<Tensor<float> *> InferenceEngine::prepareNodeInputs(
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

void InferenceEngine::applyOptimizations() { applyConstantFolding(); }

void InferenceEngine::applyConstantFolding() {
    for (auto node : graph_->getTopologicallySortedNodes()) {
        std::vector<Tensor<float> *> inputs;
        try {
            inputs = prepareNodeInputs(node);
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

void InferenceEngine::registerOperators() {
    registry_.registerOperator(
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
            return Operators<float>::gemm(A, B, bias, transA, transB, alpha,
                                          beta);
        });

    registry_.registerOperator(
        OpType::Flatten,
        [](const Node *node, const std::vector<Tensor<float> *> &inputs) {
            if (inputs.size() != 1) {
                throw std::runtime_error("Flatten operation expects 1 input");
            }
            auto axisOpt = node->getAttribute<int64_t>("axis");
            if (!axisOpt) {
                throw std::runtime_error("Axis missing for flatten operation");
            }
            return Operators<float>::flatten(*inputs[0], axisOpt.value());
        });

    registry_.registerOperator(
        OpType::Relu, []([[maybe_unused]] const Node *node,
                         const std::vector<Tensor<float> *> &inputs) {
            if (inputs.size() != 1) {
                throw std::runtime_error("Relu operation expects 1 input");
            }
            return Operators<float>::relu(*inputs[0]);
        });

    registry_.registerOperator(
        OpType::Add, []([[maybe_unused]] const Node *node,
                        const std::vector<Tensor<float> *> &inputs) {
            if (inputs.size() != 2) {
                throw std::runtime_error("Add operation expects 2 inputs");
            }
            return Operators<float>::add(*inputs[0], *inputs[1]);
        });
}
