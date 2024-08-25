#include "input_loader.h"

#include <assert.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

Tensor<float> load_input(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if(!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    // TODO: generalize, probably something like Triton's model registry with
    // config file specifying input dims.
    std::vector<unsigned char> bytes(784);
    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());

    std::vector<float> floatValues(bytes.size());
    for(size_t i = 0; i < bytes.size(); ++i) {
        floatValues[i] = static_cast<float>(bytes[i]);
    }

    assert(floatValues.size() == 784);
    return Tensor<float>{floatValues, std::vector<uint64_t>{1, 1, 28, 28}};
}
