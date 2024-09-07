
#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <yaml-cpp/yaml.h>

#include <string>
#include <unordered_map>
#include <vector>

enum class DataType {
    FLOAT32,
};

enum class Device {
    CPU,
    CUDA,
};

struct TensorConfig {
    std::string name;
    std::vector<uint64_t> shape;
    DataType data_type;
};

class ModelConfig {
   public:
    ModelConfig() = default;
    ModelConfig(const std::string& config_file);

    std::string get_model_path() const { return model_path; }
    std::string get_model_format() const { return model_format; }
    Device get_device() const { return device; }
    const std::vector<TensorConfig>& get_inputs() const { return inputs; }
    const std::vector<TensorConfig>& get_outputs() const { return outputs; }
    int get_batch_size() const { return batch_size; }

   private:
    std::string model_path;
    std::string model_format;
    Device device;
    std::vector<TensorConfig> inputs;
    std::vector<TensorConfig> outputs;
    int batch_size;

    void parse_config_file(const std::string& config_file);
    DataType string_to_data_type(const std::string& str);
    Device string_to_device(const std::string& str);
    TensorConfig parse_tensor_config(const YAML::Node& node);
};

#endif