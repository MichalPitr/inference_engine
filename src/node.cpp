#include "node.h"
#include "tensor.h"

OpType onnxOpTypeConverter(const std::string opType);

Node::Node(const std::string &name, OpType opType)
    : name(name), opType(opType) {}

Node::Node(const onnx::NodeProto &nodeProto)
{
    opType = onnxOpTypeConverter(nodeProto.op_type());

    for (const auto &input : nodeProto.input())
    {
        inputs.push_back(input);
    }

    for (const auto &output : nodeProto.output())
    {
        outputs.push_back(output);
    }
}

const std::string &Node::getName() const
{
    return name;
}

const std::vector<std::string> &Node::getInputs() const
{
    return inputs;
}

const std::vector<std::string> &Node::getOutputs() const
{
    return outputs;
}

void Node::addInput(std::string input)
{
    inputs.push_back(input);
}

void Node::addOutput(std::string output)
{
    outputs.push_back(output);
}

OpType onnxOpTypeConverter(const std::string opType)
{
    if (opType == "Gemm")
    {
        return OpType::Gemm;
    }
    else if (opType == "Relu")
    {
        return OpType::Relu;
    }
    else if (opType == "Flatten")
    {
        return OpType::Flatten;
    }
    throw std::runtime_error("Unknown operation type: " + opType);
}