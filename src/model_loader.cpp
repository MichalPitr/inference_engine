#include "model_loader.h"

#include <fstream>
#include <memory>

std::vector<float> reinterpret_string_to_float(const std::string &str);

std::unique_ptr<InferenceEngine> ModelLoader::load(const std::string &modelFile)
{
    std::cout << "model path:" << modelFile << std::endl;

    std::fstream input(modelFile, std::ios::in | std::ios::binary);
    if (!input.is_open())
    {
        throw std::runtime_error("Failed to open the ONNX model file: " + modelFile);
    }

    onnx::ModelProto model;
    if (!model.ParseFromIstream(&input))
    {
        throw std::runtime_error("Failed to parse the ONNX model");
    }

    // Validate Model (Basic)
    if (!model.has_graph() || model.graph().node_size() == 0)
    {
        throw std::runtime_error("Invalid ONNX model: missing graph or nodes");
    }

    std::unordered_map<std::string, Tensor<float>> weights;
    for (const auto &initializer : model.graph().initializer())
    {
        if (initializer.data_type() != onnx::TensorProto::FLOAT)
        {
            throw std::runtime_error("Unsupported initializer data type");
        }

        std::vector<float> data = reinterpret_string_to_float(initializer.raw_data());
        const std::vector<uint64_t> shape(initializer.dims().begin(), initializer.dims().end());
        weights[initializer.name()] = Tensor<float>{data, shape};
    }

    // Debugging Information (Optional)
    std::cout << "Loaded ONNX model: " << model.graph().name() << std::endl;
    std::cout << "Number of initializers: " << weights.size() << std::endl;

    return std::unique_ptr<InferenceEngine>(
        new InferenceEngine(model.graph(), std::move(weights)));
}

std::vector<float> reinterpret_string_to_float(const std::string &str)
{
    // Safety Check: Ensure size is a multiple of sizeof(float)
    if (str.size() % sizeof(float) != 0)
    {
        throw std::runtime_error("String size is not a multiple of sizeof(float)");
    }

    // Create a temporary buffer and copy the string's data
    std::vector<char> buffer(str.begin(), str.end());

    // Reinterpret the buffer's data as floats
    return std::vector<float>(
        reinterpret_cast<const float *>(buffer.data()),
        reinterpret_cast<const float *>(buffer.data() + str.size()));
}