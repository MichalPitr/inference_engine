#include <fstream>
#include <stdexcept>
#include <vector>

#include "tensor.h"

Tensor<float> load_input(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    constexpr size_t expected_size = 28 * 28;  // 784
    std::vector<unsigned char> bytes(expected_size);

    file.read(reinterpret_cast<char*>(bytes.data()), expected_size);

    if (file.gcount() != expected_size) {
        throw std::runtime_error("Unexpected file size: " + filename);
    }

    std::vector<float> floatValues;
    floatValues.reserve(expected_size);

    for (unsigned char byte : bytes) {
        floatValues.push_back(static_cast<float>(byte));
    }

    return Tensor<float>{std::move(floatValues), {1, 1, 28, 28}};
}