
#include "operators.h"

#include <assert.h>

#include <iostream>

#include "gemm.h"
#include "gemm_cuda.h"

// Returns a Tensor containing the result of A*B + bias
template <typename T>
Tensor<T> gemm(const Tensor<T>& A, const Tensor<T>& B, const Tensor<T>& bias,
               const bool transA, const bool transB, const float alpha,
               const float beta) {
    if(A.shape().size() != 2 || B.shape().size() != 2 ||
       bias.shape().size() == 0) {
        std::cerr << "A dims: " << A.shape().size() << " B dims "
                  << B.shape().size() << " C dims " << bias.shape().size()
                  << std::endl;
        std::cerr << "A.shape: " << A.stringShape() << std::endl;
        std::cerr << "B.shape: " << B.stringShape() << std::endl;
        std::cerr << "bias.shape: " << bias.stringShape() << std::endl;

        throw std::invalid_argument("Invalid dimensions for Gemm inputs.");
    }
    if(!transA && !transB && A.shape()[1] != B.shape()[0]) {
        std::cerr << "A.shape: " << A.stringShape() << std::endl;
        std::cerr << "B.shape: " << B.stringShape() << std::endl;
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication in Gemm.");
    }
    if(transA && !transB && A.shape()[0] != B.shape()[0]) {
        std::cerr << "A.shape: " << A.stringShape() << std::endl;
        std::cerr << "B.shape: " << B.stringShape() << std::endl;
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication in Gemm.");
    }
    if(transB && !transA && A.shape()[1] != B.shape()[1]) {
        std::cerr << "A.shape: " << A.stringShape() << std::endl;
        std::cerr << "B.shape: " << B.stringShape() << std::endl;
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication in Gemm.");
    }
    if(transA && transB && A.shape()[0] != B.shape()[1]) {
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

    // TODO: add mechanism to detect if CUDA is supported at startup.
    // Currently slower for smaller number of iterations. Probably inefficient
    // transfer?
    if(true) {
        gemm_cuda(AData, BData, BiasData, outData.data(), N, M, K, transA,
                  transB, alpha, beta);
    } else {
        gemm(AData, BData, BiasData, outData.data(), N, M, K, transA, transB,
             alpha, beta);
    }

    Tensor<T> result = Tensor<T>(outData, dims);
    return result;
}

// flatten returns a new flattened version of node. Caller is responsible for
// managing memory.
template <typename T>
Tensor<T> flatten(Tensor<T>& tensor, uint64_t axis) {
    assert(axis <= tensor.shape().size());

    uint64_t dimBefore = 1;
    for(std::size_t i = 0; i < axis; ++i) {
        dimBefore *= tensor.shape()[i];
    }

    uint64_t dimAfter = 1;
    for(std::size_t i = axis; i < tensor.shape().size(); ++i) {
        dimAfter *= tensor.shape()[i];
    }

    // copy initialize. Would be better if we could modify it in place, but we
    // don't know if some other function relies on the input tensor. If we can
    // do some dependency analysis, we could probably optimize this.
    Tensor<T> flat(tensor);
    flat.setShape({dimBefore, dimAfter});
    return flat;
}

template <typename T>
Tensor<T> relu(Tensor<T>& tensor) {
    Tensor<T> output(tensor);
    T* raw = output.raw_data();
    for(std::size_t i = 0; i < output.size(); ++i) {
        raw[i] = std::max(0.0f, raw[i]);
    }

    return output;
}

template <typename T>
Tensor<T> add(Tensor<T>& A, Tensor<T>& B) {
    assert(A.shape() == B.shape());
    Tensor<T> output(A);
    T* raw = output.raw_data();
    T* b_raw = B.raw_data();
    for(std::size_t i = 0; i < output.size(); ++i) {
        raw[i] += b_raw[i];
    }

    return output;
}

// template <typename T>
// Tensor<T> conv(Tensor<T>& tensor)
// {
//     // TODO: implement
// }

template Tensor<float> flatten<float>(Tensor<float>& tensor, uint64_t axis);
template Tensor<float> relu<float>(Tensor<float>& tensor);
template Tensor<float> add<float>(Tensor<float>& A, Tensor<float>& B);
template Tensor<float> gemm(const Tensor<float>& A, const Tensor<float>& B,
                            const Tensor<float>& bias, const bool transA,
                            const bool transB, const float alpha,
                            const float beta);
