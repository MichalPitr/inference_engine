#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstdint>

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
    T* raw_data();
    std::vector<uint64_t> shape() const;
    uint64_t size() const;

private:
    std::vector<T> data_;
    std::vector<uint64_t> shape_;
};

// ... (You can add the implementation of template functions outside the class if needed) ...

#endif // TENSOR_H