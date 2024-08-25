#ifndef TENSOR_H
#define TENSOR_H

#include <cstdint>
#include <string>
#include <vector>

template <typename T>
class Tensor {
   public:
    Tensor() = default;

    Tensor(const std::vector<T>& data, const std::vector<uint64_t>& shape);

    const std::vector<T>& data() const;
    const T* raw_data() const;
    T* raw_data();

    std::vector<uint64_t> shape() const;
    void setShape(const std::vector<uint64_t>& shape);

    std::string stringShape() const;
    std::string to_string() const;

    uint64_t size() const;

   private:
    std::vector<T> data_;
    std::vector<uint64_t> shape_;
};

#endif  // TENSOR_H