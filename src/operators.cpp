
#include "operators.h"

#include <assert.h>

#include <iostream>

#include "gemm_cpu.h"
#include "kernels.h"

//-----------------//
//  CPU operators  //
//-----------------//

template <typename T>
Tensor<T> CpuOperators<T>::gemm(const Tensor<T>& A, const Tensor<T>& B,
                                const Tensor<T>& bias, bool transA, bool transB,
                                float alpha, float beta) {
    validate_gemm_inputs(A, B, bias, transA, transB);

    // Calculate output dimensions depending on transpositions.
    uint64_t N = transA ? A.shape()[1] : A.shape()[0];
    uint64_t M = transB ? B.shape()[1] : B.shape()[0];
    uint64_t K = transB ? B.shape()[0] : B.shape()[1];
    std::vector<uint64_t> dims{N, K};

    Tensor<T> out{std::move(dims)};

    // assert(A.device() == DeviceType::CPU);
    const T* AData = A.data();
    const T* BData = B.data();
    const T* BiasData = bias.data();
    gemm_cpu(AData, BData, BiasData, out.data(), N, M, K, transA, transB, alpha,
             beta);

    return out;
}

template <typename T>
Tensor<T> CpuOperators<T>::flatten(Tensor<T>& tensor, uint64_t axis) {
    return base_flatten(tensor, axis);
}

template <typename T>
Tensor<T> CpuOperators<T>::relu(const Tensor<T>& tensor) {
    Tensor<T> output(tensor);
    T* raw = output.data();
    for (std::size_t i = 0; i < output.size(); ++i) {
        raw[i] = std::max(static_cast<T>(0), raw[i]);
    }
    return output;
}

template <typename T>
Tensor<T> CpuOperators<T>::add(const Tensor<T>& A, const Tensor<T>& B) {
    assert(A.shape() == B.shape());
    Tensor<T> output(A);
    T* raw = output.data();
    const T* b_raw = B.data();
    for (std::size_t i = 0; i < output.size(); ++i) {
        raw[i] += b_raw[i];
    }
    return output;
}

//------------------//
//  CUDA operators  //
//------------------//

template <typename T>
Tensor<T> CudaOperators<T>::gemm(const Tensor<T>& A, const Tensor<T>& B,
                                 const Tensor<T>& bias, bool transA,
                                 bool transB, float alpha, float beta) {
    validate_gemm_inputs(A, B, bias, transA, transB);

    // Calculate output dimensions depending on transpositions.
    uint64_t N = transA ? A.shape()[1] : A.shape()[0];
    uint64_t M = transB ? B.shape()[1] : B.shape()[0];
    uint64_t K = transB ? B.shape()[0] : B.shape()[1];

    std::vector<uint64_t> dims{N, K};

    Tensor<T> out{std::move(dims)};

    assert(A.device() == DeviceType::CUDA);
    const T* AData = A.data();
    const T* BData = B.data();
    const T* BiasData = bias.data();

    gemm_cuda_tiled(AData, BData, BiasData, out.data(), N, M, K, transA, transB,
                    alpha, beta);

    return out;
}

template <typename T>
Tensor<T> CudaOperators<T>::flatten(Tensor<T>& tensor, uint64_t axis) {
    return base_flatten(tensor, axis);
}

// template <typename T>
// Tensor<T> CudaOperators<T>::relu(const Tensor<T>& tensor) {
//     Tensor<T> output(tensor);
//     relu_cuda(output.data(), output.size());
//     return output;
// }

// template <typename T>
// Tensor<T> CudaOperators<T>::add(const Tensor<T>& A, const Tensor<T>& B) {
//     assert(A.shape() == B.shape());
//     Tensor<T> output(A);
//     add_cuda(output.data(), B.data(), output.size());
//     return output;
// }

//------------------//
//      shared      //
//------------------//

template <typename T>
void validate_gemm_inputs(const Tensor<T>& A, const Tensor<T>& B,
                          const Tensor<T>& bias, bool transA, bool transB) {
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
}

template <typename T>
Tensor<T> base_flatten(Tensor<T>& tensor, uint64_t axis) {
    assert(axis <= tensor.shape().size());

    uint64_t dimBefore = 1;
    for (std::size_t i = 0; i < axis; ++i) {
        dimBefore *= tensor.shape()[i];
    }

    uint64_t dimAfter = 1;
    for (std::size_t i = axis; i < tensor.shape().size(); ++i) {
        dimAfter *= tensor.shape()[i];
    }
    tensor.setShape({dimBefore, dimAfter});
    return tensor;
}

template class CpuOperators<float>;
template class CudaOperators<float>;
