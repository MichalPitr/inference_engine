#include "node.h"
#include "tensor.h"

Node::Node(const std::string& name, OpType opType)
    : name(name), opType(opType) {}

const std::string& Node::getName() const {
    return name;
}

const std::vector<std::string>& Node::getInputs() const {
    return inputs;
}

const std::vector<std::string>& Node::getOutputs() const {
    return outputs;
}

void Node::addInput(std::string input) {
    inputs.push_back(input);
}

void Node::addOutput(std::string output) {
    outputs.push_back(output);
}