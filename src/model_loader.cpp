#include "model_loader.h"

#include <fstream>
#include <memory>

#include "graph.h"
#include "node.h"

std::unique_ptr<InferenceEngine> ModelLoader::load(
    const std::string& modelFile) {
    onnx::ModelProto model;
    {
        std::ifstream input(modelFile, std::ios::binary);
        if (!input || !model.ParseFromIstream(&input)) {
            throw std::runtime_error(
                "Failed to load or parse the ONNX model: " + modelFile);
        }
    }

    if (!model.has_graph() || model.graph().node_size() == 0) {
        throw std::runtime_error("Invalid ONNX model: missing graph or nodes");
    }

    auto weights = load_weights(model);
    auto graph = std::make_unique<Graph>(model.graph());
    return std::make_unique<InferenceEngine>(std::move(graph),
                                             std::move(weights));
}

std::unordered_map<std::string, Tensor<float>> ModelLoader::load_weights(
    const onnx::ModelProto& model) {
    std::unordered_map<std::string, Tensor<float>> weights;
    for (const auto& initializer : model.graph().initializer()) {
        if (initializer.data_type() != onnx::TensorProto::FLOAT) {
            throw std::runtime_error("Unsupported initializer data type");
        }

        const auto& raw_data = initializer.raw_data();
        const float* data_ptr = reinterpret_cast<const float*>(raw_data.data());
        std::vector<float> data(data_ptr,
                                data_ptr + raw_data.size() / sizeof(float));

        std::vector<uint64_t> shape(initializer.dims().begin(),
                                    initializer.dims().end());
        weights.emplace(initializer.name(),
                        Tensor<float>{std::move(data), std::move(shape)});
    }
    return weights;
}