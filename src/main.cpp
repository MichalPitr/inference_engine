#include <iostream>
#include <fstream>
#include <random>
#include <assert.h>
#include <vector>
#include <iomanip> // For std::hex
#include <numeric> // for std::accumulate
#include <span>

#include "onnx-ml.pb.h" // Include the generated header
#include "input_loader.h"
#include "tensor.h"
#include "operators.h"

float extract_const(onnx::NodeProto node);
int getFlattenAxis(const onnx::NodeProto &node);
std::vector<float> reinterpret_string_to_float(const std::string& str);

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <input.data>" << std::endl;
        return 1;
    }

    std::string modelFile = argv[1];
    std::string inputFile = argv[2];

    std::cout << "ONNX model file: " << modelFile << std::endl;
    std::cout << "User input file: " << inputFile << std::endl;

    std::fstream input(modelFile, std::ios::in | std::ios::binary);
    if (!input.is_open())
    {
        std::cerr << "Failed to open the ONNX model file!" << std::endl;
        return -1;
    }

    onnx::ModelProto model;

    // Parse the model from the file
    if (!model.ParseFromIstream(&input))
    {
        std::cerr << "Failed to parse the ONNX model!" << std::endl;
        return -1;
    }

    const onnx::GraphProto &graph = model.graph();

    // Example: Print the model's graph input names
    std::cout << "Model Inputs:" << std::endl;
    for (const auto &input : graph.input())
    {
        std::cout << "  - " << input.name() << std::endl;
    }

    // Print graph outputs
    std::cout << "\nOutputs:\n";
    for (const auto &output : graph.output())
    {
        std::cout << "  - Name: " << output.name() << ", Type: " << output.type().denotation() << std::endl;
    }
    assert(graph.output_size() == 1);
    const std::string graph_output = graph.output()[0].name();

    // Print nodes (operators)
    std::cout << "\nNodes (Operators):\n";
    for (const auto &node : graph.node())
    {
        std::cout << "  - OpType: " << node.op_type();
        std::cout << ", Name: " << node.name();
        std::cout << ", Inputs: [";
        for (const auto &input : node.input())
        {
            std::cout << input << " ";
        }
        std::cout << "], Outputs: [";
        for (const auto &output : node.output())
        {
            std::cout << output << " ";
        }
        std::cout << "]\n";
    }
    
    std::unordered_map<std::string, Tensor<float>> weights;
    for (const auto &init : graph.initializer())
    {
        std::cout << "Adding initializer: " << init.name() << std::endl;
        std::cout << "float data size " << init.float_data().size() << std::endl;
        std::vector<float> data = reinterpret_string_to_float(init.raw_data());
        const std::vector<uint64_t> shape(init.dims().begin(), init.dims().end());
        weights[init.name()] = Tensor<float>{data, shape};
    }

    std::cout << "Reading input..." << std::endl;

    // Make mock input.
    weights["onnx::Flatten_0"] = load_input(inputFile);

    // Iterate over nodes (topologically sorted)
    std::cout << "iterating over graph" << std::endl;
    std::cout << "-----------------" << std::endl;
    std::cout << std::endl;
    for (const auto &node : graph.node())
    {

        std::cout << "Node: " << node.name() << std::endl;

        std::string op_type = node.op_type();
        std::vector<Tensor<float>> inputs{};

        std::cout << "Inputs: ";
        for (const auto &input_name : node.input())
        {

            if (weights.find(input_name) == weights.end())
            {
                throw std::runtime_error("Input not found: " + input_name);
            }
            inputs.push_back(weights[input_name]);
        }
        std::cout << std::endl;

        Tensor<float> output{};
        if (op_type == "Gemm") {
            assert(inputs.size() == 3);
            Tensor<float> A = inputs[0];
            Tensor<float> B = inputs[1];
            Tensor<float> bias = inputs[2];
            output = gemm(A, B, bias);
        } else if (op_type == "Flatten") {
            assert(inputs.size() == 1);
            Tensor<float> tensor = inputs[0];
            uint64_t axis = getFlattenAxis(node);
            output = flatten(tensor, axis);
        } else if (op_type == "Relu") {
            assert(inputs.size() == 1);
            Tensor<float> tensor = inputs[0];
            output = relu(tensor);
        } else {
            throw std::runtime_error("Op_type no supported: " + op_type);
        }

        if (output.size() != 0) {
            weights[node.output(0)] = output;
        } else {
            throw std::runtime_error("Got nullptr output after inference loop.");
        }
    }
    
    if (weights.find(graph_output) != weights.end())
    {
        std::cout << "Model output: " << std::endl;
        Tensor<float> res = weights[graph_output];
        std::vector<float> out = res.data();
        for (uint64_t i = 0; i < res.size(); ++i)
        {
            std::cout << ", " << out[i];
        }
        std::cout << std::endl;
    }

    return 0;
}

float extract_const(onnx::NodeProto node)
{
    std::cout << "Const: " << node.output(0) << std::endl;
    const onnx::AttributeProto *value_attr = nullptr;
    for (const auto &attr : node.attribute())
    {
        if (attr.name() == "value")
        {
            value_attr = &attr;
            break;
        }
    }
    if (!value_attr)
    {
        std::cerr << "Expected constant to have value attr, got nothing" << std::endl;
        exit(1);
    }
    const onnx::TensorProto &tensor = value_attr->t();
    float floatValue;
    if (tensor.data_type() != onnx::TensorProto::FLOAT)
    {
        std::cerr << "Const is not of float type" << std::endl;
        exit(1);
    }

    if (tensor.dims_size() == 0)
    {
        // Const is a scalar
        floatValue = tensor.float_data(0); // Extract the float
        std::cout << "Constant value: " << floatValue << std::endl;
        return floatValue;
    }
    std::cerr << "Unimplemented handling for multi-dim consts" << std::endl;
    exit(1);
}

int getFlattenAxis(const onnx::NodeProto &node)
{
    int axis = 1; // Default value (as per ONNX specification)

    for (const auto &attr : node.attribute())
    {
        if (attr.name() == "axis")
        {
            if (attr.has_i())
            {
                axis = attr.i(); // Get the 'axis' value
                break;
            }
            else
            {
                std::cerr << "Error: Flatten node has 'axis' attribute, but it's not an integer." << std::endl;
            }
        }
    }

    return axis;
}

std::vector<float> reinterpret_string_to_float(const std::string& str) {
    // Safety Check: Ensure size is a multiple of sizeof(float)
    if (str.size() % sizeof(float) != 0) {
        throw std::runtime_error("String size is not a multiple of sizeof(float)");
    }

    // Create a temporary buffer and copy the string's data
    std::vector<char> buffer(str.begin(), str.end());

    // Reinterpret the buffer's data as floats
    return std::vector<float>(
        reinterpret_cast<const float*>(buffer.data()), 
        reinterpret_cast<const float*>(buffer.data() + str.size())
    );
}