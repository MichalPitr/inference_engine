#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstdint>
#include <string>

template <typename T>
class Tensor {
public:
    // Default constructors
    Tensor() = default;
    Tensor(Tensor&&) = default; // move
    Tensor& operator=(Tensor&&) = default;
    Tensor(const Tensor&) = default; // copy
    Tensor& operator=(Tensor const&) = default;
    ~Tensor() = default;

    Tensor(const std::vector<T>& data, const std::vector<uint64_t>& shape);

    const std::vector<T>& data() const;
    const T* raw_data() const;
    T* raw_data();

    std::vector<uint64_t> shape() const;
    void setShape(const std::vector<uint64_t>& shape);
    std::string stringShape() const;
    
    uint64_t size() const;

private:
    std::vector<T> data_;
    std::vector<uint64_t> shape_;
};

#endif // TENSOR_H