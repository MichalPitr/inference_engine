
#include "operators.h"

#include <assert.h>

#include <iostream>

#include "gemm_cpu.h"
#include "gemm_cuda.h"

// Returns a Tensor containing the result of A*B + bias
template <typename T>
Tensor<T> Operators<T>::gemm(const Tensor<T>& A, const Tensor<T>& B,
                             const Tensor<T>& bias, bool transA, bool transB,
                             float alpha, float beta) {
    if (A.shape().size() != 2 || B.shape().size() != 2 ||
        bias.shape().size() == 0) {
        std::cerr << "A dims: " << A.shape().size() << " B dims "
                  << B.shape().size() << " C dims " << bias.shape().size()
                  << std::endl;
        std::cerr << "A.shape: " << A.stringShape() << std::endl;
        std::cerr << "B.shape: " << B.stringShape() << std::endl;
        std::cerr << "bias.shape: " << bias.stringShape() << std::endl;

        throw std::invalid_argument("Invalid dimensions for Gemm inputs.");
    }
    if (!transA && !transB && A.shape()[1] != B.shape()[0]) {
        std::cerr << "A.shape: " << A.stringShape() << std::endl;
        std::cerr << "B.shape: " << B.stringShape() << std::endl;
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication in Gemm.");
    }
    if (transA && !transB && A.shape()[0] != B.shape()[0]) {
        std::cerr << "A.shape: " << A.stringShape() << std::endl;
        std::cerr << "B.shape: " << B.stringShape() << std::endl;
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication in Gemm.");
    }
    if (transB && !transA && A.shape()[1] != B.shape()[1]) {
        std::cerr << "A.shape: " << A.stringShape() << std::endl;
        std::cerr << "B.shape: " << B.stringShape() << std::endl;
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication in Gemm.");
    }
    if (transA && transB && A.shape()[0] != B.shape()[1]) {
        std::cerr << "A.shape: " << A.stringShape() << std::endl;
        std::cerr << "B.shape: " << B.stringShape() << std::endl;
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication in Gemm.");
    }

    // Calculate output dimensions depending on transpositions.
    uint64_t N = transA ? A.shape()[1] : A.shape()[0];
    uint64_t M = transB ? B.shape()[1] : B.shape()[0];
    uint64_t K = transB ? B.shape()[0] : B.shape()[1];

    std::vector<uint64_t> dims{N, K};

    std::vector<T> outData(N * K);

    const T* AData = A.raw_data();
    const T* BData = B.raw_data();
    const T* BiasData = bias.raw_data();

    if (true) {
        gemm_cuda(AData, BData, BiasData, outData.data(), N, M, K, transA,
                  transB, alpha, beta);
    } else {
        gemm_cpu(AData, BData, BiasData, outData.data(), N, M, K, transA,
                 transB, alpha, beta);
    }

    Tensor<T> result = Tensor<T>(std::move(outData), std::move(dims));
    return result;
}

template <typename T>
Tensor<T> Operators<T>::flatten(const Tensor<T>& tensor, uint64_t axis) {
    assert(axis <= tensor.shape().size());

    uint64_t dimBefore = 1;
    for (std::size_t i = 0; i < axis; ++i) {
        dimBefore *= tensor.shape()[i];
    }

    uint64_t dimAfter = 1;
    for (std::size_t i = axis; i < tensor.shape().size(); ++i) {
        dimAfter *= tensor.shape()[i];
    }

    Tensor<T> flat(tensor);
    flat.setShape({dimBefore, dimAfter});
    return flat;
}

template <typename T>
Tensor<T> Operators<T>::relu(const Tensor<T>& tensor) {
    Tensor<T> output(tensor);
    T* raw = output.raw_data();
    for (std::size_t i = 0; i < output.size(); ++i) {
        raw[i] = std::max(static_cast<T>(0), raw[i]);
    }
    return output;
}

template <typename T>
Tensor<T> Operators<T>::add(const Tensor<T>& A, const Tensor<T>& B) {
    assert(A.shape() == B.shape());
    Tensor<T> output(A);
    T* raw = output.raw_data();
    const T* b_raw = B.raw_data();
    for (std::size_t i = 0; i < output.size(); ++i) {
        raw[i] += b_raw[i];
    }
    return output;
}

template class Operators<float>;
