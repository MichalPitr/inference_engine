#include "input_loader.h"

#include <fstream>
#include <stdexcept>
#include <vector>

Tensor<float> load_input(const std::string& filename,
                         const ModelConfig& config) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    const auto& input_shape = config.get_inputs()[0].shape;
    size_t expected_size = 1;
    for (const auto& dim : input_shape) {
        expected_size *= dim;
    }
    std::vector<unsigned char> bytes(expected_size);

    file.read(reinterpret_cast<char*>(bytes.data()), expected_size);

    if (file.gcount() != static_cast<std::streamsize>(expected_size)) {
        throw std::runtime_error("Unexpected file size: " + filename);
    }

    std::vector<float> floatValues;
    floatValues.reserve(expected_size);

    for (unsigned char byte : bytes) {
        floatValues.push_back(static_cast<float>(byte));
    }
    return Tensor<float>{floatValues.data(), input_shape};
}