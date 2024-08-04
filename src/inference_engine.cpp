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
        switch (node->getOpType())
        {
        case OpType::Gemm:
        {
            float alpha = node->getAttribute<float>("alpha").value_or(1.0);
            float beta = node->getAttribute<float>("beta").value_or(1.0);
            int transA = node->getAttribute<int64_t>("transA").value_or(0);
            int transB = node->getAttribute<int64_t>("transB").value_or(0);

            assert(inputs.size() == 3);
            Tensor<float> A = inputs[0];
            Tensor<float> B = inputs[1];
            Tensor<float> bias = inputs[2];
            output = gemm(A, B, bias, transA, transB, alpha, beta);
            break;
        }
        case OpType::Flatten:
        {
            assert(inputs.size() == 1);
            Tensor<float> tensor = inputs[0];
            auto axisOpt = node->getAttribute<int64_t>("axis");
            if (!axisOpt.has_value())
                throw std::runtime_error("Axis missing for flatten operation");
            output = flatten(tensor, axisOpt.value());
            break;
        }
        case OpType::Relu:
        {
            assert(inputs.size() == 1);
            output = relu(inputs[0]);
            break;
        }
        case OpType::Add:
        {
            assert(inputs.size() == 2);
            output = add(inputs[0], inputs[1]);
            break;
        }
        case OpType::Conv:
        {
            assert(inputs.size() == 3);
            Tensor<float> X = inputs[0];
            Tensor<float> W = inputs[1];
            Tensor<float> B = inputs[2];
            auto dilation = node->getAttribute<std::vector<int64_t>>("dilations");
            if (!dilation.has_value())
                throw std::runtime_error("dilations missing for conv operator");
            auto kernel_shape = node->getAttribute<std::vector<int64_t>>("kernel_shape");
            if (!kernel_shape.has_value())
                throw std::runtime_error("kernel shape missing for conv operator");
            auto pads = node->getAttribute<std::vector<int64_t>>("kernel_shape");
            if (!pads.has_value())
                throw std::runtime_error("pads missing for conv operator");
            auto strides = node->getAttribute<std::vector<int64_t>>("strides");
            if (!strides.has_value())
                throw std::runtime_error("strides missing for conv operator");
            auto group = node->getAttribute<int64_t>("group");
            if (!group.has_value())
                throw std::runtime_error("group missing for conv operator");
                
            throw std::logic_error("Not Implemented");
        }
        default:
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