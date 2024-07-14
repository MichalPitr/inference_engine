#include "inference_engine.h"

#include <stdexcept>
#include <iostream>

#include "operators.h"
#include "onnx_helper.h"

InferenceEngine::InferenceEngine(const onnx::GraphProto &graph,
                                 std::unordered_map<std::string, Tensor<float>> weights)
    : graph_(graph), weights_(std::move(weights))
{
}

Tensor<float> InferenceEngine::infer(const Tensor<float> &input)
{
    weights_[graph_.input(0).name()] = input; // Assume single input for now

    // Iterate over nodes (assuming topologically sorted)
    for (const auto &node : graph_.node())
    {
        std::string op_type = node.op_type();
        std::vector<Tensor<float>> inputs;
        for (const auto &input_name : node.input())
        {
            if (weights_.find(input_name) == weights_.end())
            {
                throw std::runtime_error("Input not found: " + input_name);
            }
            inputs.push_back(weights_[input_name]);
        }

        Tensor<float> output{};
        if (op_type == "Gemm")
        {
            bool alphaOk, betaOk, transAOk, transBOk;
            float alphaVal, betaVal;
            int transAVal, transBVal;

            std::tie(alphaOk, alphaVal) = getAttr<float>(node, "alpha");
            if (!alphaOk)
                alphaVal = 1.0;
            std::tie(betaOk, betaVal) = getAttr<float>(node, "beta");
            if (!betaOk)
                alphaVal = 1.0;
            std::tie(transAOk, transAVal) = getAttr<int>(node, "transA");
            if (!transAOk)
                transAVal = false;
            std::tie(transBOk, transBVal) = getAttr<int>(node, "transB");
            if (!transBOk)
                transBVal = false;
            std::cout << "alpha: " << alphaOk << ", " << alphaVal << "\n";
            std::cout << "beta: " << betaOk << ", " << betaVal << "\n";
            std::cout << "transA: " << transAOk << ", " << transAVal << "\n";
            std::cout << "transB: " << transBOk << ", " << transBVal << "\n";

            assert(inputs.size() == 3);
            Tensor<float> A = inputs[0];
            Tensor<float> B = inputs[1];
            Tensor<float> bias = inputs[2];
            output = gemm(A, B, bias, transAVal, transBVal, alphaVal, betaVal);
        }
        else if (op_type == "Flatten")
        {
            assert(inputs.size() == 1);
            Tensor<float> tensor = inputs[0];
            bool axisOk;
            int axis;
            std::tie(axisOk, axis) = getAttr<int>(node, "axis");
            if (!axisOk)
                throw std::runtime_error("Axis missing for flatten operation");
            output = flatten(tensor, axis);
        }
        else if (op_type == "Relu")
        {
            assert(inputs.size() == 1);
            Tensor<float> tensor = inputs[0];
            output = relu(tensor);
        }
        else
        {
            throw std::runtime_error("Op_type no supported: " + op_type);
        }

        if (output.size() != 0)
        {
            weights_[node.output(0)] = output;
        }
        else
        {
            throw std::runtime_error("Got nullptr output after inference loop.");
        }

        weights_[node.output(0)] = output;
    }

    // Get output
    std::string graph_output = graph_.output(0).name();
    if (weights_.find(graph_output) == weights_.end())
    {
        throw std::runtime_error("Output not found: " + graph_output);
    }

    return weights_[graph_output];
}
