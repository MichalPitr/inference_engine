#ifndef OPTYPES_H
#define OPTYPES_H

enum class OpType {
    Input,     // Input to the graph
    Output,    // Output of the graph
    Add,       // Add two tensors
    Gemm,      // General Matrix Multiplication
    Flatten,   // Flatten an input
    Relu,      // Rectified Linear Unit
    Conv,      // Convolutional Layer
    MaxPool,   // Max Pooling Layer
    Constant,  // Constant node
};

#endif