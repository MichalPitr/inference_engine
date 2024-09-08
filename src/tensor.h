#ifndef TENSOR_H
#define TENSOR_H

#include <cstdint>
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
    void to(DeviceType newDevice);

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

   private:
    T* data_;
    std::vector<size_t> shape_;
    size_t size_;
    std::shared_ptr<Allocator> allocator_;

    void allocateAndCopy(const T* data);
    void freeMemory();
};

#endif  // TENSOR_H