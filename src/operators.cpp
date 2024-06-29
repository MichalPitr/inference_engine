
#include <iostream>
#include <assert.h>

#include "gemm.h"
#include "operators.h"

// Returns a Tensor containing the result of A*B + bias
template <typename T>
Tensor<T> gemm(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& bias) {
    std::cout << "Op: Gemm" << std::endl;

    // Input Validation
    if (A.shape().size() != 2 || B.shape().size() != 2 || bias.shape().size() == 0) {
        std::cerr << "A dims: " << A.shape().size() << " B dims " << B.shape().size() << " C dims " << bias.shape().size() << std::endl;
        std::cerr << "A.shape: " << A.stringShape() << std::endl;
        std::cerr << "B.shape: " << B.stringShape() << std::endl;
        std::cerr << "bias.shape: " << bias.stringShape() << std::endl;

        throw std::invalid_argument("Invalid dimensions for Gemm inputs.");
    }
    if (A.shape()[1] != B.shape()[0]) {
        std::cerr << "A.shape: " << A.stringShape() << std::endl;
        std::cerr << "B.shape: " << B.stringShape() << std::endl;
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication in Gemm.");
    }
    if (B.shape()[0] != bias.shape()[0]) {
        std::cerr << "A.shape: " << A.stringShape() << std::endl;
        std::cerr << "B.shape: " << B.stringShape() << std::endl;
        std::cerr << "bias.shape: " << bias.stringShape() << std::endl;
        throw std::invalid_argument("Bias dimensions are not compatible with the result in Gemm.");
    }

    std::cout << "A.shape = (" << A.shape()[0] << ", " << A.shape()[1] << ")" << std::endl;
    std::cout << "B.shape = (" << B.shape()[0] << ", " << B.shape()[1] << ")" << std::endl;
    std::cout << "bias.shape = (" << bias.shape()[0] << ", " << bias.shape()[1] << ")"<< std::endl;

    // Calculate output dimensions
    uint64_t N = A.shape()[0];
    uint64_t M = B.shape()[0];
    uint64_t K = B.shape()[1];
    std::cout << "N: " << N << std::endl;
    std::cout << "M: " << M << std::endl;
    std::cout << "K: " << K << std::endl;

    std::vector<uint64_t> dims {N, K};

    std::cout << "shape(" << N <<", " << K << ")\n";

    // Allocate memory for output and copy bias (C) using a loop
    std::vector<T> outData(N*K);
    std::cout << "outData size " << outData.size() << std::endl;

    // Perform GEMM operation
    // Pass raw pointers to the underlying `gemm` function
    const T* AData = A.raw_data();
    const T* BData = B.raw_data();
    const T* BiasData = bias.raw_data();

    std::cout << "Running gemm" << std::endl;
    gemm(AData, BData, BiasData, outData.data(), N, M, K);
    std::cout << "finished gemm" << std::endl;

    Tensor<T> result = Tensor<T>(outData, dims);

    // Print out values
    std::cout << "out: ";
    for (std::size_t i = 0; i < outData.size(); ++i) {
        std::cout << outData[i] << ", ";
    }
    std::cout << std::endl;

    return result;
}

// flatten returns a new flattened version of node. Caller is responsible for managing memory.
template <typename T>
Tensor<T> flatten(Tensor<T> &tensor, uint64_t axis)
{
    std::cout << "Op: Flatten" << std::endl;
    assert(axis <= tensor.shape().size());

    uint64_t dimBefore = 1;
    for (std::size_t i = 0; i < axis; ++i)
    {
        dimBefore *= tensor.shape()[i];
    }
    std::cout << "dim before: " << dimBefore << std::endl;

    uint64_t dimAfter = 1;
    for (std::size_t i = axis; i < tensor.shape().size(); ++i)
    {
        dimAfter *= tensor.shape()[i];
    }
    std::cout << "dim after: " << dimAfter << std::endl;

    // copy initialize. Would be better if we could modify it in place, but we
    // don't know if some other function relies on the input tensor. If we can do some dependency analysis, we could
    // probably optimize this.
    Tensor<T> flat(tensor);
    for (auto s : flat.shape())
    {
        std::cout << "shape dim: " << s << '\n';
    }
    flat.setShape({dimBefore, dimAfter});
    for (auto s : flat.shape())
    {
        std::cout << "shape after dim: " << s << '\n';
    }
    // Diagnostic printing
    std::cout << "flatten out:";
    for (std::size_t i = 0; i < flat.size(); ++i)
    {
        std::cout << " " << flat.data()[i];
    }
    std::cout << std::endl;

    // Set tensor name.
    return flat;
}

template <typename T>
Tensor<T> relu(Tensor<T>& tensor)
{
    std::cout << "Op: Relu" << std::endl;

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
template Tensor<float> gemm(const Tensor<float>& A, const Tensor<float>& B, const Tensor<float>& bias);
