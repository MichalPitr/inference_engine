#ifndef NODE_H
#define NODE_H

#include <string>

#include "tensor.h"

enum class OpType
{
    Input,   // Input to the graph
    Output,  // Output of the graph
    Gemm,    // General Matrix Multiplication
    Flatten, // Flatten an input
    Relu,    // Rectified Linear Unit
};

class Node {
public:
    Node(const std::string& name, const OpType optype, Tensor* tensor); // Constructor

    // Accessors (getters)
    const std::string& getName() const;
    Tensor* getTensor() const; 
    const std::vector<Node*>& getInputs() const;
    const std::vector<Node*>& getOutputs() const;

    // Modifiers (setters/adders)
    void addInput(Node* node);
    void addOutput(Node* node);

private:
    std::string name;
    OpType opType;
    Tensor* tensor; 
    std::vector<Node*> inputs;
    std::vector<Node*> outputs;
};

#endif