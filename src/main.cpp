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
#include "gemm.h"

Tensor *relu(std::vector<const Tensor*> &inputs);
Tensor *flatten(std::vector<const Tensor*> &inputs, uint64_t axis);
Tensor* gemm(std::vector<const Tensor*>& inputs);
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

    std::map<std::string, Tensor> weights;
    for (const auto &init : graph.initializer())
    {
        std::cout << "Adding initializer: " << init.name() << std::endl;
        std::cout << "float data size " << init.float_data().size() << std::endl;
        std::vector<float> data = reinterpret_string_to_float(init.raw_data());
        const std::vector<uint64_t> shape(init.dims().begin(), init.dims().end());
        weights[init.name()] = Tensor(data, shape, DataType::FLOAT32);
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
        std::vector<const Tensor *> inputs;

        std::cout << "Inputs: ";
        for (const auto &input_name : node.input())
        {

            if (weights.find(input_name) == weights.end())
            {
                throw std::runtime_error("Input not found: " + input_name);
            }
            inputs.push_back(&weights[input_name]);
        }
        std::cout << std::endl;

        Tensor* output = nullptr;
        if (op_type == "Gemm") {
            output = gemm(inputs);
        } else if (op_type == "Flatten") {
            uint64_t axis = getFlattenAxis(node);
            output = flatten(inputs, axis);
        } else if (op_type == "Relu") {
            output = relu(inputs);
        } else {
            throw std::runtime_error("Op_type no supported: " + op_type);
        }

        if (output != nullptr) {
            weights[node.output(0)] = *output;
            delete output;
        } else {
            throw std::runtime_error("Got nullptr output after inference loop.");
        }
    }
    
    if (weights.find(graph_output) != weights.end())
    {
        std::cout << "Model output: " << std::endl;
        Tensor res = weights[graph_output];
        std::vector<float> out = res.getData();
        for (uint64_t i = 0; i < res.getNumElements(); ++i)
        {
            std::cout << ", " << out[i];
        }
        std::cout << std::endl;
    }

    return 0;
}

Tensor* gemm(std::vector<const Tensor*> &inputs) {
    std::cout << "Op: Gemm" << std::endl;

    if (inputs.size() != 3) {
        throw std::runtime_error("Gemm operator expects exactly three input tensors.");
    }

    const auto* A = inputs[0];
    const auto* B = inputs[1];
    const auto* C = inputs[2];

    // Input Validation
    if (A->getShape().size() != 2 || B->getShape().size() != 2 || C->getShape().size() != 1) {
        std::cerr << "A dims: " << A->getShape().size() << "B dims" << B->getShape().size() << "C dims" << C->getShape().size() << std::endl;
        throw std::invalid_argument("Invalid dimensions for Gemm inputs.");
    }
    if (A->getShape()[1] != B->getShape()[1]) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication in Gemm.");
    }
    if (B->getShape()[0] != C->getShape()[0]) {
        throw std::invalid_argument("Bias dimensions are not compatible with the result in Gemm.");
    }

    std::cout << "A.shape = (" << A->getShape()[0] << ", " << A->getShape()[1] << ")" << std::endl;
    std::cout << "B.shape = (" << B->getShape()[0] << ", " << B->getShape()[1] << ")" << std::endl;
    std::cout << "C.shape = (" << C->getShape()[0] << ")" << std::endl;


    // Calculate output dimensions
    uint64_t N = A->getShape()[1];
    uint64_t M = B->getShape()[0];
    uint64_t K = A->getShape()[0];
    std::cout << "N: " << N << std::endl;
    std::cout << "M: " << M << std::endl;
    std::cout << "K: " << K << std::endl;


    std::vector<uint64_t> dims = {K, M};

    // Allocate memory for output and copy bias (C) using a loop
    std::vector<float> outData(M);
    std::cout << "outData size " << outData.size() << std::endl;

    // Perform GEMM operation
    // Pass raw pointers to the underlying `gemm` function
    const float* AData = reinterpret_cast<const float*>(A->getData().data());
    const float* BData = reinterpret_cast<const float*>(B->getData().data());
    const float* CData = reinterpret_cast<const float*>(C->getData().data());

    std::cout << "Running gemm" << std::endl;
    gemm(BData, AData, CData, outData.data(), M, N, K); // Assuming your gemm function is modified to accept raw pointers
    std::cout << "finished gemm" << std::endl;

    Tensor* result = new Tensor(outData, dims, DataType::FLOAT32);

    // Print out values
    std::cout << "out: ";
    for (std::size_t i = 0; i < outData.size(); ++i) {
        std::cout << outData[i] << ", ";
    }
    std::cout << std::endl;

    return result;
}

// flatten returns a new flattened version of node. Caller is responsible for managing memory.
Tensor *flatten(std::vector<const Tensor*> &inputs, uint64_t axis)
{
    std::cout << "Op: Flatten" << std::endl;
    if (inputs.size() != 1)
    {
        throw std::runtime_error("Expected inputs.size to be 1 in flatten operation.");
    }

    const auto inputTensor = inputs[0];

    assert(inputTensor->getShape().size() == 4);
    assert(axis <= inputTensor->getShape().size());
    
    // Mnist input only
    assert(inputTensor->getShape()[0] == 1); // batch size
    assert(inputTensor->getShape()[1] == 1); // channels
    assert(inputTensor->getShape()[2] == 28); 
    assert(inputTensor->getShape()[3] == 28);

    uint64_t dimBefore = 1;
    for (std::size_t i = 0; i < axis; ++i) {
        dimBefore *= inputTensor->getShape()[i];
    }
    std::cout << "dim before: " << dimBefore << std::endl;

    uint64_t dimAfter = 1;
    for (std::size_t i = axis; i < inputTensor->getShape().size(); ++i) {
        dimAfter *= inputTensor->getShape()[i];
    }
    std::cout << "dim after: " << dimAfter << std::endl;
    
    std::vector<float> flattened(inputTensor->getData()); 
    Tensor* flat = new Tensor(flattened, std::vector<uint64_t> {dimBefore, dimAfter}, DataType::FLOAT32);
    
    // Diagnostic printing
    std::cout << "flatten.size(): " << flattened.size() << std::endl;
    std::cout << "flatten out:";
    for (std::size_t i = 0; i < flattened.size(); ++i)
    {
        std::cout << " " << flattened[i];
    }
    std::cout << std::endl;

    // Set tensor name.
    return flat;
}

// relu
Tensor* relu(std::vector<const Tensor*> &inputs)
{
    std::cout << "Op: Relu" << std::endl;
    assert(inputs.size() == 1);
    const auto &inputTensor = *inputs[0];

    if (inputTensor.getDataType() == DataType::FLOAT32 && !inputTensor.getData().empty())
    {
        // Copy input data.
        std::vector<float> outputData(inputTensor.getData());
        for (std::size_t i = 0; i < outputData.size(); ++i)
        {
            outputData[i] = std::max(0.0f, outputData[i]);
        }

        // Print the modified output values (optional)
        std::cout << "ReLU: ";
        assert(inputTensor.getShape().size() == 2);
        for (uint64_t i = 0; i < inputTensor.getShape()[1]; ++i)
        { // Assuming the second dimension is the relevant one
            std::cout << outputData[i] << ", ";
        }
        std::cout << std::endl;
        return new Tensor(outputData, inputTensor.getShape(), DataType::FLOAT32);
    }
    else
    {
        std::cerr << "Unsupported data type or empty raw data in ReLU" << std::endl;
        exit(1);
    }
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