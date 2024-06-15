#include <stdexcept>
#include "tensor.h"

Tensor::Tensor(const std::vector<float>& data, const std::vector<uint64_t>& shape, DataType dataType)
    : data_(data), shape_(shape), dataType_(dataType) {
        if (dataType != DataType::FLOAT32) {
            throw std::runtime_error("Unsupported data type in Tensor constructor.");
        }
        uint64_t expectedSize = 1;
        for (uint64_t dim : shape) {
            expectedSize *= dim;
        }
        if (data.size() != expectedSize) {
            throw std::runtime_error("Data size does not match the expected size.");
        }
    }

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) { // Check for self-assignment
        data_ = other.data_;
        shape_ = other.shape_;
        dataType_ = other.dataType_;
    }
    return *this;
}

const std::vector<float>& Tensor::getData() const {
    return data_;
}

std::vector<uint64_t> Tensor::getShape() const {
    return shape_;
}

DataType Tensor::getDataType() const {
    return dataType_;
}

std::size_t Tensor::getNumElements() const {
    return data_.size();
}