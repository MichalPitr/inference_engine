
// model_config.cpp
#include "model_config.h"

#include <iostream>
#include <stdexcept>

ModelConfig::ModelConfig(const std::string& config_file) {
    parse_config_file(config_file);
}

void ModelConfig::parse_config_file(const std::string& config_file) {
    try {
        YAML::Node config = YAML::LoadFile(config_file);

        model_path = config["model_path"].as<std::string>();
        model_format = config["model_format"].as<std::string>();
        execution_provider = string_to_execution_provider(
            config["execution_provider"].as<std::string>());

        batch_size = config["batch_size"].as<int>(1);

        if (config["inputs"]) {
            for (const auto& input : config["inputs"]) {
                inputs.push_back(parse_tensor_config(input));
            }
        }

        if (config["outputs"]) {
            for (const auto& output : config["outputs"]) {
                outputs.push_back(parse_tensor_config(output));
            }
        }

    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Error parsing YAML config file: " +
                                 std::string(e.what()));
    }

    // Validate that all required fields are set
    if (model_path.empty() || model_format.empty() || inputs.empty() ||
        outputs.empty()) {
        throw std::runtime_error(
            "Invalid config file: missing required fields");
    }
}

DataType ModelConfig::string_to_data_type(const std::string& str) {
    if (str == "FLOAT32") return DataType::FLOAT32;
    throw std::runtime_error("Unknown data type: " + str);
}

ExecutionProvider ModelConfig::string_to_execution_provider(
    const std::string& str) {
    if (str == "CPU") return ExecutionProvider::CPU;
    if (str == "CUDA") return ExecutionProvider::CUDA;
    throw std::runtime_error("Unknown execution provider: " + str);
}

TensorConfig ModelConfig::parse_tensor_config(const YAML::Node& node) {
    TensorConfig tensor_config;
    tensor_config.name = node["name"].as<std::string>();
    tensor_config.shape = node["shape"].as<std::vector<int64_t>>();
    tensor_config.data_type =
        string_to_data_type(node["data_type"].as<std::string>());
    return tensor_config;
}