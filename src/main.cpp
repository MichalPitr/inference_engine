#include <iostream>
#include <fstream>
#include <random>
#include <assert.h>
#include <vector>
#include <iomanip> // For std::hex
#include <numeric> // for std::accumulate

#include "onnx-ml.pb.h" // Include the generated header

void gemm(const float *A, const float *B, const float *C, float *output, const int n, const int m, const int k);
onnx::TensorProto *relu(std::vector<const onnx::TensorProto *> &inputs, const onnx::NodeProto &node);
onnx::TensorProto *flatten(std::vector<const onnx::TensorProto *> &inputs, const onnx::NodeProto &node);
onnx::TensorProto *create_mock_input();
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

    // Make mock input.
    onnx::TensorProto *modelInput = create_mock_input();
    weights[modelInput->name()] = modelInput;

    // Iterate over nodes (topologically sorted)
    std::cout << "iterating over graph" << std::endl;
    std::cout << "-----------------" << std::endl;
    std::cout << std::endl;
    for (const auto &node : graph.node())
    {
        
        std::cout << "Node: " << node.name() << std::endl;

        std::string op_type = node.op_type();
        std::vector<const onnx::TensorProto *> inputs;

        std::cout << "Inputs: ";
        for (const std::string input_name : node.input())
        {

            if (weights.find(input_name) == weights.end())
            {
                std::cerr << "Input: " << input_name << " not in weights. Aborting." << std::endl;
                exit(1);
            }
            const onnx::TensorProto *in = weights[input_name];
            std::cout << in->name() << ", ";
            inputs.push_back(in);
        }
        std::cout << std::endl;

        if (op_type == "Gemm")
        {
            // Convert inputs to raw arrays
            assert(inputs.size() == 3);

            // A shape is (1, 784)
            assert(inputs[0]->dims_size() == 2);
            std::string A_raw = inputs[0]->raw_data();
            int A_size = A_raw.size() / sizeof(float);
            std::cout << "inputs[0]->dims(1): " << inputs[0]->dims(1) << std::endl;
            
            // dims(0) gives batch size, dims(1) starts with data shape.
            std::cout << "A_raw.size(): " << A_raw.size() << std::endl;
            std::cout << "A_size: " << A_size << std::endl;
            assert(A_size == inputs[0]->dims(1));
            float *A = new float[A_size];
            std::memcpy(A, A_raw.data(), A_raw.size());

            // B shape is (512, 784)
            assert(inputs[1]->dims_size() == 2);
            std::string B_raw = inputs[1]->raw_data();
            int B_size = B_raw.size() / sizeof(float);
            assert(B_size == inputs[1]->dims(0) * inputs[1]->dims(1));
            std::cout << "B_raw.size(): " << B_raw.size() << std::endl;
            std::cout << "B_size: " << B_size << std::endl;
            float *B = new float[B_size];
            std::memcpy(B, B_raw.data(), B_raw.size());

            // C shape is (512)
            assert(inputs[2]->dims_size() == 1);
            std::string C_raw = inputs[2]->raw_data();
            int C_size = C_raw.size() / sizeof(float);
            assert(C_size == inputs[2]->dims(0));
            std::cout << "C_raw.size(): " << C_raw.size() << std::endl;
            std::cout << "C_size: " << C_size << std::endl;
            float *C = new float[C_size];
            std::memcpy(C, C_raw.data(), C_raw.size());

            // out shape is (512)
            float *out = new float[C_size];
            std::cout << "out_size: " << sizeof(out) << std::endl;

            // Pass to gemm
            // Save output to node_outputs.
            // gemm();
            // TODO: currently GEMM supports only A*B + C type of operations, A*B is assumed to be matrix * vector product.
            // A = (1, 784)
            // B = (512, 784)
            // C = (512)
            // has to be B * A + C
            // Where n = 512, m = 784, k = 1

            // Actual setting is A * B^T + C = (1, 784) * (784, 512) + (512) = (512) + (512) = (512)...
            // A * B^T = B * A^T = (512, 784) * (784, 1) = (512, 1).
            gemm(B, A, C, out, inputs[0]->dims(0), inputs[1]->dims(0), 1);
            onnx::TensorProto *result = new onnx::TensorProto;
            result->set_data_type(onnx::TensorProto::FLOAT);

            // Set dimensions.
            result->add_dims(inputs[0]->dims(0));
            result->add_dims(inputs[2]->dims(0));
            result->set_raw_data(out, sizeof(float) * C_size);

            assert(node.output_size() == 1);
            std::string out_name = node.output()[0];
            result->set_name(out_name.c_str()); // Set the input name (important for ONNX runtimes)
            weights[out_name] = result;
        }
        else if (op_type == "Constant")
        {
            float constant = extract_const(node);
        }
        else if (op_type == "Flatten")
        {
            onnx::TensorProto *output = flatten(inputs, node);
            weights[output->name()] = output;
        }
        else if (op_type == "Relu")
        {
            onnx::TensorProto *output = relu(inputs, node);
            weights[output->name()] = output;
        }
        else
        {
            std::cerr << "Unsupported operation: " << op_type << ". Skipping..." << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}

onnx::TensorProto *create_mock_input()
{
    onnx::TensorProto *modelInput = new onnx::TensorProto;
    modelInput->set_name("onnx::Flatten_0"); // Set the input name (important for ONNX runtimes)

    // Set data type to FLOAT
    modelInput->set_data_type(onnx::TensorProto::FLOAT);

    // Set dimensions to [1, 1, 28, 28]
    modelInput->add_dims(1);
    modelInput->add_dims(1);
    modelInput->add_dims(28);
    modelInput->add_dims(28);

    // Fill with 1s (assuming FLOAT data type)
    std::vector<float> data(1 * 1 * 28 * 28, 1.0f); // 784 elements
    modelInput->set_raw_data(data.data(), data.size() * sizeof(float));
    return modelInput;
}

// flatten returns a new flattened version of node. Caller is responsible for managing memory.
onnx::TensorProto *flatten(std::vector<const onnx::TensorProto *> &inputs, const onnx::NodeProto &node)
{
    std::cout << "Op: Flatten" << std::endl;
    assert(inputs.size() == 1);

    onnx::TensorProto *flattened = new onnx::TensorProto;
    flattened->set_data_type(onnx::TensorProto::FLOAT);

    // Get output dims
    const onnx::TensorProto *tensor = inputs[0];
    int64_t axis = getFlattenAxis(node);
    int64_t dimBefore = std::accumulate(tensor->dims().begin(), tensor->dims().begin() + axis, 1, std::multiplies<int64_t>());
    int64_t dimAfter = std::accumulate(tensor->dims().begin() + axis, tensor->dims().end(), 1, std::multiplies<int64_t>());

    // Set dimensions.
    flattened->add_dims(dimBefore);
    flattened->add_dims(dimAfter);

    flattened->set_raw_data(tensor->raw_data());

    assert(node.output_size() == 1);
    std::string out_name = node.output()[0];
    flattened->set_name(out_name.c_str()); // Set the input name (important for ONNX runtimes)
    std::cout << "Flattened name: " << flattened->name() << std::endl;
    return flattened;
}

// relu
onnx::TensorProto *relu(std::vector<const onnx::TensorProto *> &inputs, const onnx::NodeProto &node)
{
    std::cout << "Op: Relu" << std::endl;
    assert(inputs.size() == 1);

    onnx::TensorProto *out = new onnx::TensorProto;
    out->set_data_type(onnx::TensorProto::FLOAT);

    // Get output dims
    const onnx::TensorProto *tensor = inputs[0];

    // Set dimensions.
    out->add_dims(tensor->dims(0));
    out->add_dims(tensor->dims(1));

    std::string raw = inputs[0]->raw_data();
    int raw_size = raw.size();
    std::cout << "raw_size: " << raw_size << std::endl;
    float A[raw_size];

    std::cout << "raw.size(): " << raw.size() << std::endl;
    std::memcpy(A, raw.data(), raw.size());

    for (int i = 0; i < raw_size; ++i) {
        if (A[i] < 0) {
            A[i] = 0;
        }
    }

    out->set_raw_data(A, raw_size);
    assert(node.output_size() == 1);
    std::string out_name = node.output()[0];
    out->set_name(out_name.c_str()); // Set the input name (important for ONNX runtimes)
    std::cout << "relu name: " << out->name() << std::endl;
    return out;
}

// gemm returns C = A * B
// A is (n, m)
// B is (m, k)
// C is (n, k)
void gemm(const float *A, const float *B, const float *C, float *output, const int n, const int m, const int k)
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

    assert(k == 1);
    for (int i = 0; i < n; ++i)
    {
        output[i] += C[i];
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