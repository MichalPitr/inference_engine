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
    Node(const std::string& name, const OpType optype); // Constructor

    // Accessors (getters)
    const std::string& getName() const;
    const std::vector<std::string>& getInputs() const;
    const std::vector<std::string>& getOutputs() const;

    // Modifiers (setters/adders)
    void addInput(std::string input);
    void addOutput(std::string output);

private:
    std::string name;
    OpType opType;
    // Sorted list of inputs as expected the by the corresponding opType.
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

#endif