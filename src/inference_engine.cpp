#include "inference_engine.h"

#include <stdexcept>
#include <iostream>

#include "operators.h"
#include "onnx_helper.h"

std::string op_type_to_string(OpType op_type);

InferenceEngine::InferenceEngine(std::unique_ptr<Graph> graph,
                                 std::unordered_map<std::string, Tensor<float>> weights)
    : graph_(std::move(graph)), weights_(std::move(weights))
{
}

Tensor<float> InferenceEngine::infer(const Tensor<float> &input)
{
    weights_[graph_->getInputName(0)] = input;

    // Iterate over nodes (assuming topologically sorted)
    for (const auto &node : graph_->getTopologicallySortedNodes())
    {
        std::vector<Tensor<float>> inputs;
        for (const auto &input_name : node->getInputs())
        {
            if (weights_.find(input_name) == weights_.end())
            {
                throw std::runtime_error("Input not found: " + input_name);
            }
            inputs.push_back(weights_[input_name]);
        }

        Tensor<float> output{};
        const auto op_type = node->getOpType();
        if (op_type == OpType::Gemm)
        {
            bool alphaOk, betaOk, transAOk, transBOk;
            float alphaVal, betaVal;
            int transAVal, transBVal;

            std::tie(alphaOk, alphaVal) = node->getAttribute<float>("alpha");
            if (!alphaOk)
                alphaVal = 1.0;
            std::tie(betaOk, betaVal) = node->getAttribute<float>("beta");
            if (!betaOk)
                alphaVal = 1.0;
            std::tie(transAOk, transAVal) = node->getAttribute<int64_t>("transA");
            if (!transAOk)
                transAVal = false;
            std::tie(transBOk, transBVal) = node->getAttribute<int64_t>("transB");
            if (!transBOk)
                transBVal = false;

            assert(inputs.size() == 3);
            Tensor<float> A = inputs[0];
            Tensor<float> B = inputs[1];
            Tensor<float> bias = inputs[2];
            output = gemm(A, B, bias, transAVal, transBVal, alphaVal, betaVal);
        }
        else if (op_type == OpType::Flatten)
        {
            assert(inputs.size() == 1);
            Tensor<float> tensor = inputs[0];
            bool axisOk;
            int axis;
            std::tie(axisOk, axis) = node->getAttribute<int64_t>("axis");
            if (!axisOk)
                throw std::runtime_error("Axis missing for flatten operation");
            output = flatten(tensor, axis);
        }
        else if (op_type == OpType::Relu)
        {
            assert(inputs.size() == 1);
            Tensor<float> tensor = inputs[0];
            output = relu(tensor);
        }
        else
        {
            throw std::runtime_error("Op_type no supported: " + op_type_to_string(op_type));
        }

        if (output.size() != 0)
        {
            weights_[node->getOutputs()[0]] = output;
        }
        else
        {
            throw std::runtime_error("Got nullptr output after inference loop.");
        }

        weights_[node->getOutputs()[0]] = output;
    }

    // Get output
    std::string graph_output = graph_->getOutputName(0);
    if (weights_.find(graph_output) == weights_.end())
    {
        throw std::runtime_error("Output not found: " + graph_output);
    }

    return weights_[graph_output];
}

std::string op_type_to_string(OpType op_type)
{
    switch (op_type)
    {
    case OpType::Input:
        return "Input";
    case OpType::Output:
        return "Output";
    case OpType::Gemm:
        return "Gemm";
    case OpType::Flatten:
        return "Flatten";
    case OpType::Relu:
        return "Relu";
    default:
        throw std::runtime_error("Unknown op_type");
    }
}