#include "tensor.h"

#include <stdexcept>
#include <numeric>

template <typename T>
Tensor<T>::Tensor(const std::vector<T>& data, const std::vector<uint64_t>& shape)
    : data_(data), shape_(shape) {
    
    uint64_t expectedSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint64_t>());
    if (data_.size() != expectedSize) {
        throw std::invalid_argument("Data size does not match the specified shape.");
    }
}

template <typename T>
const std::vector<T>& Tensor<T>::data() const {
    return data_;
}

template <typename T>
const T* Tensor<T>::raw_data() const {
    return data_.data();
}

template <typename T>
T* Tensor<T>::raw_data() {
    return data_.data();
}


template <typename T>
std::vector<uint64_t> Tensor<T>::shape() const {
    return shape_;
}

template <typename T>
uint64_t Tensor<T>::size() const {
    return data_.size();
}

template <typename T>
void Tensor<T>::setShape(const std::vector<uint64_t>& shape) {
    shape_ = shape;
}

template class Tensor<float>;
