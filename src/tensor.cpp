#include "tensor.h"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "cpu_allocator.h"
#include "cuda_allocator.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape,
                  std::shared_ptr<Allocator> alloc)
    : data_(nullptr),
      shape_(shape),
      size_(std::accumulate(shape.begin(), shape.end(), 1ULL,
                            std::multiplies<size_t>())),
      allocator_(std::move(alloc)) {
    allocateAndCopy(nullptr);
}

template <typename T>
Tensor<T>::Tensor(const T* data, const std::vector<size_t>& shape,
                  std::shared_ptr<Allocator> alloc)
    : data_(nullptr),
      shape_(shape),
      size_(std::accumulate(shape.begin(), shape.end(), 1ULL,
                            std::multiplies<size_t>())),
      allocator_(std::move(alloc)) {
    allocateAndCopy(data);
}

template <typename T>
Tensor<T>::Tensor(const Tensor& other)
    : data_(nullptr),
      shape_(other.shape_),
      size_(other.size_),
      allocator_(other.allocator_) {
    allocateAndCopy(other.data_);
}

template <typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : data_(other.data_),
      shape_(std::move(other.shape_)),
      size_(other.size_),
      allocator_(std::move(other.allocator_)) {
    other.data_ = nullptr;
    other.size_ = 0;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& other) {
    if (this != &other) {
        freeMemory();
        data_ = nullptr;
        shape_ = other.shape_;
        size_ = other.size_;
        allocator_ = other.allocator_;
        allocateAndCopy(other.data_);
    }
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        freeMemory();
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        allocator_ = std::move(other.allocator_);
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

template <typename T>
Tensor<T>::~Tensor() {
    freeMemory();
}

template <typename T>
void Tensor<T>::allocateAndCopy(const T* src) {
    data_ = static_cast<T*>(allocator_->allocate(sizeof(T) * size_));

    if (src) {
        if (allocator_->getDeviceType() == DeviceType::CPU) {
            std::memcpy(data_, src, size_ * sizeof(T));
        }
#ifdef USE_CUDA
        else if (allocator_->getDeviceType() == DeviceType::CUDA) {
            cudaMemcpy(data_, src, size_ * sizeof(T), cudaMemcpyHostToDevice);
        }
#endif
        else {
            throw std::runtime_error("Unsupported device type");
        }
    }
}

template <typename T>
void Tensor<T>::freeMemory() {
    if (data_) {
        allocator_->deallocate(data_);
        data_ = nullptr;
    }
}

template <typename T>
void Tensor<T>::to(DeviceType newDevice,
                   std::shared_ptr<Allocator> newAllocator) {
    if (newDevice == allocator_->getDeviceType()) return;

    if (newAllocator->getDeviceType() != newDevice) {
        throw std::runtime_error(
            "Provided allocator does not match the requested device type");
    }

    T* newData = static_cast<T*>(newAllocator->allocate(sizeof(T) * size_));

    if (allocator_->getDeviceType() == DeviceType::CPU &&
        newDevice == DeviceType::CUDA) {
        cudaMemcpy(newData, data_, size_ * sizeof(T), cudaMemcpyHostToDevice);
    } else if (allocator_->getDeviceType() == DeviceType::CUDA &&
               newDevice == DeviceType::CPU) {
        cudaMemcpy(newData, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
    } else {
        throw std::runtime_error("Unsupported device transition");
    }

    allocator_->deallocate(data_);
    data_ = newData;
    allocator_ = newAllocator;
}

template <typename T>
void Tensor<T>::setShape(const std::vector<size_t>& newShape) {
    size_t size = std::accumulate(newShape.begin(), newShape.end(), 1ULL,
                                  std::multiplies<size_t>());
    if (size != size_) {
        throw std::runtime_error("Expected setShape to match current shape.");
    }
    shape_ = std::move(newShape);
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

    if (device() == DeviceType::CPU) {
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