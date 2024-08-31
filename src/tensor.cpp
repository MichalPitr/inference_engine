#include "tensor.h"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

template <typename T>
Tensor<T>::Tensor(const std::vector<uint64_t>& shape, DeviceType device)
    : shape_(shape), device_(device) {
    size_ = std::accumulate(shape.begin(), shape.end(), 1ULL,
                            std::multiplies<uint64_t>());
    allocateMemory();
}

template <typename T>
Tensor<T>::Tensor(const std::vector<T>& data,
                  const std::vector<uint64_t>& shape, DeviceType device)
    : shape_(shape), device_(device) {
    size_ = std::accumulate(shape.begin(), shape.end(), 1ULL,
                            std::multiplies<uint64_t>());
    if (data.size() != size_) {
        throw std::invalid_argument(
            "Data size does not match the specified shape.");
    }
    allocateMemory();
    copyFrom(data.data(), size_);
}

template <typename T>
Tensor<T>::Tensor(const T* data, const std::vector<uint64_t>& shape,
                  DeviceType device)
    : shape_(shape), device_(device) {
    size_ = std::accumulate(shape.begin(), shape.end(), 1ULL,
                            std::multiplies<uint64_t>());
    allocateMemory();
    copyFrom(data, size_);
}

template <typename T>
Tensor<T>::Tensor(const Tensor& other)
    : shape_(other.shape_), size_(other.size_), device_(other.device_) {
    allocateMemory();
    copyFrom(other);
}

template <typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : data_(other.data_),
      shape_(std::move(other.shape_)),
      size_(other.size_),
      device_(other.device_) {
    other.data_ = nullptr;
}

template <typename T>
Tensor<T>::~Tensor() {
    freeMemory();
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        freeMemory();
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        device_ = other.device_;
        other.data_ = nullptr;
    }
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other) {
    if (this != &other) {
        freeMemory();
        shape_ = other.shape_;
        size_ = other.size_;
        device_ = other.device_;
        allocateMemory();
        copyFrom(other);
    }
    return *this;
}

template <typename T>
void Tensor<T>::copyFrom(const Tensor& other) {
    if (device_ == DeviceType::CPU && other.device_ == DeviceType::CPU) {
        std::memcpy(data_, other.data_, size_ * sizeof(T));
    }
#ifdef USE_CUDA
    else if (device_ == DeviceType::CUDA && other.device_ == DeviceType::CUDA) {
        cudaMemcpy(data_, other.data_, size_ * sizeof(T),
                   cudaMemcpyDeviceToDevice);
    } else if (device_ == DeviceType::CPU &&
               other.device_ == DeviceType::CUDA) {
        cudaMemcpy(data_, other.data_, size_ * sizeof(T),
                   cudaMemcpyDeviceToHost);
    } else if (device_ == DeviceType::CUDA &&
               other.device_ == DeviceType::CPU) {
        cudaMemcpy(data_, other.data_, size_ * sizeof(T),
                   cudaMemcpyHostToDevice);
    }
#else
    else if (device_ == DeviceType::CUDA || other.device_ == DeviceType::CUDA) {
        throw std::runtime_error("CUDA support not enabled.");
    }
#endif
}

template <typename T>
void Tensor<T>::copyFrom(const T* data, uint64_t count) {
    if (device_ == DeviceType::CPU) {
        std::memcpy(data_, data, count * sizeof(T));
    }
#ifdef USE_CUDA
    else if (device_ == DeviceType::CUDA) {
        cudaMemcpy(data_, data, count * sizeof(T), cudaMemcpyHostToDevice);
    }
#else
    else {
        throw std::runtime_error("CUDA support not enabled.");
    }
#endif
}

template <typename T>
void Tensor<T>::allocateMemory() {
    if (device_ == DeviceType::CPU) {
        data_ = new T[size_];
    } else {
#ifdef USE_CUDA
        cudaMalloc(&data_, size_ * sizeof(T));
#else
        throw std::runtime_error("CUDA support not enabled.");
#endif
    }
}

template <typename T>
void Tensor<T>::freeMemory() {
    if (data_) {
        if (device_ == DeviceType::CPU) {
            delete[] data_;
        } else {
#ifdef USE_CUDA
            cudaFree(data_);
#else
            throw std::runtime_error("CUDA support not enabled.");
#endif
        }
        data_ = nullptr;
    }
}

template <typename T>
void Tensor<T>::to(DeviceType newDevice) {
    if (newDevice == device_) return;

#ifdef USE_CUDA
    T* newData;
    if (newDevice == DeviceType::CPU) {
        newData = new T[size_];
        cudaMemcpy(newData, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
    } else {
        cudaMalloc(&newData, size_ * sizeof(T));
        cudaMemcpy(newData, data_, size_ * sizeof(T), cudaMemcpyHostToDevice);
    }

    freeMemory();
    data_ = newData;
    device_ = newDevice;
#else
    throw std::runtime_error("CUDA support not enabled.");
#endif
}

template <typename T>
std::string Tensor<T>::stringShape() const {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape_.size(); ++i) {
        oss << shape_[i];
        if (i < shape_.size() - 1) oss << ", ";
    }
    oss << ")";
    return oss.str();
}

template <typename T>
void Tensor<T>::printShape() const {
    std::cout << "Tensor Shape: " << stringShape() << std::endl;
}

template <typename T>
std::string Tensor<T>::toString() const {
    std::ostringstream oss;
    oss << "Tensor(" << stringShape() << ") [";

    if (device_ == DeviceType::CPU) {
        for (uint64_t i = 0; i < size_; ++i) {
            oss << std::setprecision(4) << data_[i];
            if (i < size_ - 1) oss << ", ";
            if (i > 0 && i % 10 == 0) oss << "\n ";
        }
    } else {
        oss << "GPU data";
    }

    oss << "]";
    return oss.str();
}

template <typename T>
void Tensor<T>::print() const {
    std::cout << toString() << std::endl;
}

// Explicit instantiation for float
template class Tensor<float>;