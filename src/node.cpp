#include "node.h"
#include "tensor.h"

Node::Node(const std::string& name, OpType opType, Tensor* tensor)
    : name(name), opType(opType), tensor(tensor) {}

const std::string& Node::getName() const {
    return name;
}

Tensor* Node::getTensor() const {
    return tensor;
}

const std::vector<Node*>& Node::getInputs() const {
    return inputs;
}

const std::vector<Node*>& Node::getOutputs() const {
    return outputs;
}

void Node::addInput(Node* node) {
    inputs.push_back(node);
}

void Node::addOutput(Node* node) {
    outputs.push_back(node);
}