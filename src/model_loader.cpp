#include "model_loader.h"

#include <fstream>
#include <memory>

#include "graph.h"
#include "node.h"

std::unique_ptr<InferenceEngine> ModelLoader::load(const ModelConfig& config) {
    onnx::ModelProto model;
    {
        std::ifstream input(config.get_model_path(), std::ios::binary);
        if (!input || !model.ParseFromIstream(&input)) {
            throw std::runtime_error(
                "Failed to load or parse the ONNX model: " +
                config.get_model_path());
        }
    }

    if (!model.has_graph() || model.graph().node_size() == 0) {
        throw std::runtime_error("Invalid ONNX model: missing graph or nodes");
    }

    validate_model(model, config);

    auto weights = load_weights(model, config);
    auto graph = std::make_unique<Graph>(model.graph());
    return std::make_unique<InferenceEngine>(std::move(graph),
                                             std::move(weights));
}

void ModelLoader::validate_model(const onnx::ModelProto& model,
                                 const ModelConfig& config) {
    if ((std::size_t)model.graph().input_size() != config.get_inputs().size()) {
        throw std::runtime_error(
            "Mismatch in number of inputs between model and config");
    }
    for (int i = 0; i < model.graph().input_size(); ++i) {
        const auto& model_input = model.graph().input(i);
        const auto& config_input = config.get_inputs()[i];
        if (model_input.name() != config_input.name) {
            throw std::runtime_error(
                "Mismatch in input names between model and config. Got: " +
                model_input.name() + ", but expected: " + config_input.name);
        }
    }

    if ((std::size_t)model.graph().output_size() !=
        config.get_outputs().size()) {
        throw std::runtime_error(
            "Mismatch in number of outputs between model and config");
    }

    for (int i = 0; i < model.graph().output_size(); ++i) {
        const auto& model_output = model.graph().output(i);
        const auto& config_output = config.get_outputs()[i];
        if (model_output.name() != config_output.name) {
            throw std::runtime_error(
                "Mismatch in output names between model and config. Got: " +
                model_output.name() + ", but expected: " + config_output.name);
        }
    }
}

std::unordered_map<std::string, Tensor<float>> ModelLoader::load_weights(
    const onnx::ModelProto& model, const ModelConfig& config) {
    std::unordered_map<std::string, Tensor<float>> weights;
    for (const auto& initializer : model.graph().initializer()) {
        if (initializer.data_type() != onnx::TensorProto::FLOAT) {
            throw std::runtime_error("Unsupported initializer data type");
        }

        const auto& raw_data = initializer.raw_data();
        const float* data_ptr = reinterpret_cast<const float*>(raw_data.data());

        std::vector<uint64_t> shape(initializer.dims().begin(),
                                    initializer.dims().end());

        DeviceType device =
            (config.get_execution_provider() == ExecutionProvider::CUDA)
                ? DeviceType::CUDA
                : DeviceType::CPU;

        weights.emplace(initializer.name(),
                        Tensor<float>{data_ptr, std::move(shape), device});
    }
    return weights;
}