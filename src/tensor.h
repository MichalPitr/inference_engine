#ifndef TENSOR_H
#define TENSOR_H

#include <cstdint>
#include <string>
#include <vector>

enum class DeviceType { CPU, CUDA };

template <typename T>
class Tensor {
   public:
    Tensor() = default;
    Tensor(const std::vector<uint64_t>& shape,
           DeviceType device = DeviceType::CPU);
    Tensor(const std::vector<T>& data, const std::vector<uint64_t>& shape,
           DeviceType device = DeviceType::CPU);
    Tensor(const T* data, const std::vector<uint64_t>& shape,
           DeviceType device = DeviceType::CPU);
    ~Tensor();

    // Allow copy constructor.
    // Copy operations
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    // Enable move operations
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    const T* data() const { return data_; }
    T* data() { return data_; }
    const std::vector<uint64_t>& shape() const { return shape_; }
    void setShape(const std::vector<uint64_t>& shape) { shape_ = shape; }
    uint64_t size() const { return size_; }
    DeviceType device() const { return device_; }

    std::string stringShape() const;
    void printShape() const;
    std::string toString() const;
    void print() const;

    void to(DeviceType newDevice);

   private:
    uint64_t size_;
    DeviceType device_;
    T* data_ = nullptr;
    std::vector<uint64_t> shape_;

    void allocateMemory();
    void freeMemory();
    void copyFrom(const Tensor& other);
    void copyFrom(const T* data, uint64_t count);
};

#endif  // TENSOR_H