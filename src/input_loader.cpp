#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <assert.h>

#include "input_loader.h"

Tensor<float> load_input(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    std::vector<unsigned char> bytes(784); // Preallocate for 784 bytes
    file.read(reinterpret_cast<char *>(bytes.data()), bytes.size());

    std::vector<float> floatValues(bytes.size());
    for (size_t i = 0; i < bytes.size(); ++i)
    {
        floatValues[i] = static_cast<float>(bytes[i]); // Direct conversion and normalize.
    }

    assert(floatValues.size() == 784);
    // Construct and return Tensor
    return Tensor<float>{floatValues, std::vector<uint64_t>{1, 1, 28, 28}}; 
}
