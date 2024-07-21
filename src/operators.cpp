
#include <iostream>
#include <assert.h>

#include "gemm.h"
#include "operators.h"

// Returns a Tensor containing the result of A*B + bias
template <typename T>
Tensor<T> gemm(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& bias, 
    const bool transA, const bool transB, const float alpha, const float beta) 
{
    // Input Validation
    if (A.shape().size() != 2 || B.shape().size() != 2 || bias.shape().size() == 0) {
        std::cerr << "A dims: " << A.shape().size() << " B dims " << B.shape().size() << " C dims " << bias.shape().size() << std::endl;
        std::cerr << "A.shape: " << A.stringShape() << std::endl;
        std::cerr << "B.shape: " << B.stringShape() << std::endl;
        std::cerr << "bias.shape: " << bias.stringShape() << std::endl;

        throw std::invalid_argument("Invalid dimensions for Gemm inputs.");
    }
    if (!transA && !transB && A.shape()[1] != B.shape()[0]) {
        std::cerr << "A.shape: " << A.stringShape() << std::endl;
        std::cerr << "B.shape: " << B.stringShape() << std::endl;
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication in Gemm.");
    }
    if (transA && !transB && A.shape()[0] != B.shape()[0]) {
        std::cerr << "A.shape: " << A.stringShape() << std::endl;
        std::cerr << "B.shape: " << B.stringShape() << std::endl;
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication in Gemm.");
    }
    if (transB && !transA && A.shape()[1] != B.shape()[1]) {
        std::cerr << "A.shape: " << A.stringShape() << std::endl;
        std::cerr << "B.shape: " << B.stringShape() << std::endl;
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication in Gemm.");
    }
    if (transA && transB && A.shape()[0] != B.shape()[1]) {
        std::cerr << "A.shape: " << A.stringShape() << std::endl;
        std::cerr << "B.shape: " << B.stringShape() << std::endl;
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication in Gemm.");
    }

    // Calculate output dimensions depending on transpositions.
    uint64_t N = transA ? A.shape()[1] : A.shape()[0];
    uint64_t M = transB ? B.shape()[1] : B.shape()[0];
    uint64_t K = transB ? B.shape()[0] : B.shape()[1];

    std::vector<uint64_t> dims {N, K};

    // Allocate memory for output and copy bias (C) using a loop
    std::vector<T> outData(N*K);

    // Perform GEMM operation
    // Pass raw pointers to the underlying `gemm` function
    const T* AData = A.raw_data();
    const T* BData = B.raw_data();
    const T* BiasData = bias.raw_data();

    gemm(AData, BData, BiasData, outData.data(), N, M, K, transA, transB, alpha, beta);
    Tensor<T> result = Tensor<T>(outData, dims);
    return result;
}

// flatten returns a new flattened version of node. Caller is responsible for managing memory.
template <typename T>
Tensor<T> flatten(Tensor<T> &tensor, uint64_t axis)
{
    assert(axis <= tensor.shape().size());

    uint64_t dimBefore = 1;
    for (std::size_t i = 0; i < axis; ++i)
    {
        dimBefore *= tensor.shape()[i];
    }

    uint64_t dimAfter = 1;
    for (std::size_t i = axis; i < tensor.shape().size(); ++i)
    {
        dimAfter *= tensor.shape()[i];
    }

    // copy initialize. Would be better if we could modify it in place, but we
    // don't know if some other function relies on the input tensor. If we can do some dependency analysis, we could
    // probably optimize this.
    Tensor<T> flat(tensor);
    flat.setShape({dimBefore, dimAfter});
    return flat;
}

template <typename T>
Tensor<T> relu(Tensor<T>& tensor)
{
    // Copy input data.
    Tensor<T> output(tensor);
    T* raw = output.raw_data();
    for (std::size_t i = 0; i < output.size(); ++i)
    {
        raw[i] = std::max(0.0f, raw[i]);
    }

    return output;
}

template Tensor<float> flatten<float>(Tensor<float> &tensor, uint64_t axis);
template Tensor<float> relu<float>(Tensor<float> &tensor);
template Tensor<float> gemm(const Tensor<float>& A, const Tensor<float>& B, const Tensor<float>& bias, const bool transA, const bool transB, const float alpha, const float beta);
