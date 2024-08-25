#include "model_loader.h"

#include <fstream>
#include <memory>

#include "graph.h"
#include "node.h"

std::vector<float> reinterpret_string_to_float(const std::string &str);

std::unique_ptr<InferenceEngine> ModelLoader::load(
    const std::string &modelFile) {
    std::ifstream input(modelFile, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open the ONNX model file: " +
                                 modelFile);
    }

    onnx::ModelProto model;
    if (!model.ParseFromIstream(&input)) {
        throw std::runtime_error("Failed to parse the ONNX model");
    }

    validate_model(model);

    std::unordered_map<std::string, Tensor<float>> weights =
        load_weights(model);

    std::unique_ptr<Graph> graph = std::make_unique<Graph>(model.graph());
    return std::make_unique<InferenceEngine>(std::move(graph),
                                             std::move(weights));
}

void ModelLoader::validate_model(const onnx::ModelProto &model) {
    if (!model.has_graph() || model.graph().node_size() == 0) {
        throw std::runtime_error("Invalid ONNX model: missing graph or nodes");
    }
}

std::unordered_map<std::string, Tensor<float>> ModelLoader::load_weights(
    const onnx::ModelProto &model) {
    std::unordered_map<std::string, Tensor<float>> weights;
    for (const auto &initializer : model.graph().initializer()) {
        if (initializer.data_type() != onnx::TensorProto::FLOAT) {
            throw std::runtime_error("Unsupported initializer data type");
        }

        std::vector<float> data =
            reinterpret_string_to_float(initializer.raw_data());
        const std::vector<uint64_t> shape(initializer.dims().begin(),
                                          initializer.dims().end());
        weights.emplace(initializer.name(),
                        Tensor<float>{std::move(data), shape});
    }
    return weights;
}

std::vector<float> reinterpret_string_to_float(const std::string &str) {
    if (str.size() % sizeof(float) != 0) {
        throw std::runtime_error(
            "String size is not a multiple of sizeof(float)");
    }

    // Directly cast the string data without copying to a buffer
    return std::vector<float>(
        reinterpret_cast<const float *>(str.data()),
        reinterpret_cast<const float *>(str.data() + str.size()));
}
