#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstdint> // For uint64_t (used for dimensions)

template <typename T>
class Tensor {
public:
    // // Default constructor
    Tensor() = default;

    Tensor(const std::vector<T>& data, const std::vector<uint64_t>& shape);

    // Accessors (const versions to prevent accidental modification)
    const std::vector<T>& data() const;
    std::vector<uint64_t> shape() const;
    uint64_t size() const;

private:
    std::vector<T> data_;
    std::vector<uint64_t> shape_;
};

// ... (You can add the implementation of template functions outside the class if needed) ...

#endif // TENSOR_H