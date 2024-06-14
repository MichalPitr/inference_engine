#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstdint> // For uint64_t (used for dimensions)

enum class DataType {
    UNKNOWN = 0,  // For error/uninitialized states
    FLOAT32, 
    // ... Add more types as needed (INT32, INT64, etc.)
};

class Tensor {
public:
    // Constructor 
    Tensor(const std::vector<float>& data, const std::vector<uint64_t>& shape, DataType dataType);

    // Accessors (const versions to prevent accidental modification)
    const std::vector<float>& getData() const;
    std::vector<uint64_t> getShape() const;
    DataType getDataType() const;
    uint64_t getNumElements() const;

private:
    std::vector<float> data_;
    std::vector<uint64_t> shape_;
    DataType dataType_;
};

// ... (You can add the implementation of template functions outside the class if needed) ...

#endif // TENSOR_H