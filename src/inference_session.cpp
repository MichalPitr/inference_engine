#include "inference_session.h"

#include <fstream>

#include "device.h"

void validate_model(const onnx::ModelProto& model, const ModelConfig& config);
std::unordered_map<std::string, Tensor<float>> load_weights(
    const onnx::ModelProto& model);

void InferenceSession::set_execution_provider(
    std::unique_ptr<ExecutionProvider> engine) {
    engine_ = std::move(engine);
}

void InferenceSession::initialize_provider() {
    pinned_allocator_ = std::make_shared<PinnedCpuAllocator>();
    engine_->transferWeightsToDevice(weights_);
}

void InferenceSession::set_input(const std::string& name,
                                 Tensor<float>& input) {
    weights_.insert_or_assign(name, input);
}

Tensor<float> InferenceSession::get_output(const std::string& name) {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("Output not found: " + name);
    }
    auto res = std::move(it->second);
    weights_.erase(it);
    res.to(DeviceType::CPU, pinned_allocator_);
    return res;
}

void InferenceSession::run() {
    for (const auto node : graph_->getTopologicallySortedNodes()) {
        auto inputs = prepare_node_inputs(node);
        auto output = engine_->evaluateNode(node, inputs);

        if (output.size() != 0) {
            weights_.insert_or_assign(node->getOutputs()[0], std::move(output));
        } else {
            throw std::runtime_error("Got empty output after inference loop.");
        }
    }
}

std::vector<Tensor<float>*> InferenceSession::prepare_node_inputs(
    const Node* node) {
    std::vector<Tensor<float>*> inputs;
    const auto& input_names = node->getInputs();
    inputs.reserve(input_names.size());

    for (const auto& input_name : input_names) {
        auto it = weights_.find(input_name);
        if (it == weights_.end()) {
            throw std::runtime_error("Input not found: " + input_name);
        }
        inputs.push_back(&it->second);
    }
    return inputs;
}

void InferenceSession::load_model(const ModelConfig& config) {
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

    weights_ = load_weights(model);
    graph_ = std::make_unique<Graph>(model.graph());
}

void validate_model(const onnx::ModelProto& model, const ModelConfig& config) {
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

std::unordered_map<std::string, Tensor<float>> load_weights(
    const onnx::ModelProto& model) {
    std::unordered_map<std::string, Tensor<float>> weights;
    for (const auto& initializer : model.graph().initializer()) {
        if (initializer.data_type() != onnx::TensorProto::FLOAT) {
            throw std::runtime_error("Unsupported initializer data type");
        }

        const auto& raw_data = initializer.raw_data();
        const float* data_ptr = reinterpret_cast<const float*>(raw_data.data());

        std::vector<uint64_t> shape(initializer.dims().begin(),
                                    initializer.dims().end());

        weights.emplace(initializer.name(),
                        Tensor<float>{data_ptr, std::move(shape)});
    }
    return weights;
}