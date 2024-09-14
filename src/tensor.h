#ifndef TENSOR_H
#define TENSOR_H

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "cpu_allocator.h"
#include "device.h"
#include "memory_allocator.h"

template <typename T>
class Tensor {
   public:
    // Constructor with shape and optional allocator
    // Constructor with shape and optional allocator
    Tensor(const std::vector<size_t>& shape,
           std::shared_ptr<Allocator> alloc = std::make_shared<CpuAllocator>());

    // Constructor with data, shape, and optional allocator
    Tensor(const T* data, const std::vector<size_t>& shape,
           std::shared_ptr<Allocator> alloc = std::make_shared<CpuAllocator>());

    // Copy constructor
    Tensor(const Tensor& other);

    // Move constructor
    Tensor(Tensor&& other) noexcept;

    // Copy assignment operator
    Tensor& operator=(const Tensor& other);

    // Move assignment operator
    Tensor& operator=(Tensor&& other) noexcept;

    // Destructor
    ~Tensor();

    // Method to move tensor to another device
    void to(DeviceType newDevice, std::shared_ptr<Allocator> newAllocator =
                                      std::make_shared<CpuAllocator>());

    // Modifications
    void setShape(const std::vector<size_t>& newShape);

    // Getter methods
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return size_; }
    DeviceType device() const { return allocator_->getDeviceType(); }
    T* data() { return data_; }
    const T* data() const { return data_; }

    // Print methods
    std::string stringShape() const;
    void printShape() const;
    std::string toString() const;
    void print() const;

    static Tensor stack(const std::vector<Tensor<T>>& tensors) {
        if (tensors.empty()) {
            throw std::invalid_argument("Cannot stack empty vector of tensors");
        }

        // Check if all tensors have the same shape
        const auto& firstShape = tensors[0].shape();
        for (size_t i = 1; i < tensors.size(); ++i) {
            if (tensors[i].shape() != firstShape) {
                throw std::invalid_argument(
                    "All tensors must have the same shape");
            }
        }

        // Calculate new shape
        std::vector<size_t> newShape = firstShape;
        newShape[0] = tensors.size() * firstShape[0];

        // Create new tensor with the calculated shape
        Tensor<T> result(newShape, tensors[0].allocator_);

        // Copy data from input tensors to the new tensor. Only supports
        // stacking on CPU.
        size_t offset = 0;
        for (const auto& tensor : tensors) {
            std::memcpy(result.data_ + offset, tensor.data_,
                        tensor.size_ * sizeof(T));
            offset += tensor.size_;
        }

        return result;
    }

   private:
    T* data_;
    std::vector<size_t> shape_;
    size_t size_;
    std::shared_ptr<Allocator> allocator_;

    void allocateAndCopy(const T* data);
    void freeMemory();
};

#endif  // TENSOR_H