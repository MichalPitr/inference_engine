#include <iostream>
#include <fstream>
#include <random>
#include <assert.h>
#include <vector>
#include <iomanip> // For std::hex
#include <numeric> // for std::accumulate

#include "onnx-ml.pb.h" // Include the generated header

void gemm(const float *A, const float *B, float *output, const int n, const int m, const int k);
void flatten(float *input, const std::vector<int> &input_dims, std::vector<int> &output_dims);
float extract_const(onnx::NodeProto node);
void printRawData(const onnx::TensorProto &tensor);
int getFlattenAxis(const onnx::NodeProto &node);

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

    std::map<std::string, const onnx::TensorProto *> weights;
    for (const auto &init : graph.initializer())
    {
        std::cout << "Adding initializer: " << init.name() << std::endl;
        weights[init.name()] = &init;
    }

    onnx::TensorProto modelInput;
    modelInput.set_name("onnx::Flatten_0"); // Set the input name (important for ONNX runtimes)

    // Set data type to FLOAT
    modelInput.set_data_type(onnx::TensorProto::FLOAT);

    // Set dimensions to [1, 1, 28, 28]
    modelInput.add_dims(1);
    modelInput.add_dims(1);
    modelInput.add_dims(28);
    modelInput.add_dims(28);

    // Fill with 1s (assuming FLOAT data type)
    std::vector<float> data(1 * 1 * 28 * 28, 1.0f); // 784 elements
    modelInput.set_raw_data(data.data(), data.size() * sizeof(float));

    weights[modelInput.name()] = &modelInput;

    // Iterate over nodes (topologically sorted)
    std::cout << "iterating over graph" << std::endl;
    for (const auto &node : graph.node())
    {

        std::cout << "Preparing inputs" << std::endl;
        std::string op_type = node.op_type();

        std::cout << "Values in the weights:\n";
        for (const auto &pair : weights)
        {
            std::cout << pair.first << std::endl; // pair.first is the key
        }

        std::vector<const onnx::TensorProto*> inputs;
        for (const std::string input_name : node.input())
        {
            std::cout << "Input: " << input_name << std::endl;

            if (weights.find(input_name) == weights.end())
            {
                std::cerr << "Input: " << input_name << " not in weights. Aborting." << std::endl;
                exit(1);
            }
            const onnx::TensorProto *in = weights[input_name];
            std::cout << "name: " << in->name() << std::endl;
            inputs.push_back(in);
        }

        // Implementations for other operations
        if (op_type == "Gemm")
        {
            // Convert inputs to raw arrays
            std::cout << "Gemm(";
            assert(inputs.size() == 3);

            for (const auto input : inputs)
            {
                std::cout << input->name() << ", ";
                const std::string &rawData = input->raw_data();
                const int numElements = rawData.size() / sizeof(float); // Calculate the number of floats
                float *weightData = new float[numElements];
                std::memcpy(weightData, rawData.data(), rawData.size());
            }
            std::cout << ");" << std::endl;
            // Pass to gemm
            // Save output to node_outputs.
            // gemm();
        }
        else if (op_type == "Constant")
        {
            float constant = extract_const(node);
        }
        else if (op_type == "Flatten")
        {
            std::cout << "Op: Flatten" << std::endl;
            // TODO: implement flatten op
            assert(inputs.size() == 1);
            assert(node.output_size() == 1);

            onnx::TensorProto *flattened = new onnx::TensorProto;
            // Set data type to FLOAT
            flattened->set_data_type(onnx::TensorProto::FLOAT);

            // Get output dims
            const onnx::TensorProto* tensor = inputs[0];
            int64_t axis = getFlattenAxis(node);
            int64_t dimBefore = std::accumulate(tensor->dims().begin(), tensor->dims().begin() + axis, 1, std::multiplies<int64_t>());
            int64_t dimAfter = std::accumulate(tensor->dims().begin() + axis, tensor->dims().end(), 1, std::multiplies<int64_t>());

            // Set dimensions to [1, 1, 28, 28]
            flattened->add_dims(dimBefore);
            flattened->add_dims(dimAfter);

            // Fill with 1s (assuming FLOAT data type)
            
            flattened->set_raw_data(tensor->raw_data());

            std::string out_name = node.output()[0];
            flattened->set_name(out_name.c_str()); // Set the input name (important for ONNX runtimes)
            std::cout << "Flattened name: " << flattened->name() << std::endl;
            weights[out_name] = flattened;
            std::cout << "Flattened name: " << weights[out_name]->name() << std::endl;
        }
        else
        {
            std::cerr << "Unsupported operation: " << op_type << ". Skipping..." << std::endl;
        }
    }

    return 0;
}

void flatten(float *input, const std::vector<int> &input_dims, std::vector<int> &output_dims)
{
    // Does nothing
}

// gemm returns C = A * B
// A is (n, m)
// B is (m, k)
// C is (n, k)
void gemm(const float *A, const float *B, float *output, const int n, const int m, const int k)
{
    for (int r = 0; r < m; ++r)
    {
        for (int c = 0; c < k; ++c)
        {
            float res = 0;
            for (int i = 0; i < n; ++i)
            {
                res += A[r * n + i] * B[i * k + c];
            }
            output[r * k + c] = res;
        }
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

void printRawData(const onnx::TensorProto &tensor)
{
    std::cout << "Raw data: ";

    switch (tensor.data_type())
    {
    case onnx::TensorProto::FLOAT:
        for (int i = 0; i < tensor.float_data_size(); ++i)
        {
            std::cout << std::fixed << tensor.float_data(i) << " ";
        }
        break;

    case onnx::TensorProto::INT32:
        for (int i = 0; i < tensor.int32_data_size(); ++i)
        {
            std::cout << tensor.int32_data(i) << " ";
        }
        break;

    case onnx::TensorProto::STRING:
        for (int i = 0; i < tensor.string_data_size(); ++i)
        {
            std::cout << tensor.string_data(i) << " ";
        }
        break;

    default:
        // Print raw bytes in hexadecimal for other data types
        const std::string &rawData = tensor.raw_data();
        for (char byte : rawData)
        {
            std::cout << std::hex << std::setw(2) << std::setfill('0')
                      << static_cast<int>(static_cast<unsigned char>(byte)) << " ";
        }
    }

    std::cout << std::endl; // Add a newline
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
                // Handle this error as you see fit
            }
        }
    }

    return axis;
}