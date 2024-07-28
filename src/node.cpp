#include "node.h"
#include "tensor.h"

OpType onnxOpTypeConverter(const std::string opType);

Node::Node(const std::string &name, OpType opType)
    : name(name), opType(opType) {}

Node::Node(const onnx::NodeProto &nodeProto)
    : name(nodeProto.name()), opType(onnxOpTypeConverter(nodeProto.op_type())),
      inputs(nodeProto.input().begin(), nodeProto.input().end()),
      outputs(nodeProto.output().begin(), nodeProto.output().end())
{
    for (const auto &attrProto : nodeProto.attribute())
    {
        attributes.emplace(attrProto.name(), Attribute(attrProto));
    }
}

const std::string &Node::getName() const
{
    return name;
}

OpType Node::getOpType() const
{
    return opType;
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

template <typename T>
std::optional<T> Node::getAttribute(const std::string &name) const
{
    auto it = attributes.find(name);
    if (it != attributes.end() && std::holds_alternative<T>(it->second.getValue())) 
    {
        return std::get<T>(it->second.getValue());
    } 
    return std::nullopt;
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
    else if (opType == "Conv")
    {
        return OpType::Conv;
    }
    else if (opType == "MaxPool")
    {
        return OpType::MaxPool;
    }
    
    throw std::runtime_error("Unknown operation type: " + opType);
}

template std::optional<int64_t> Node::getAttribute(const std::string &) const;
template std::optional<float> Node::getAttribute(const std::string &) const;
template std::optional<std::vector<int64_t>> Node::getAttribute(const std::string &) const;