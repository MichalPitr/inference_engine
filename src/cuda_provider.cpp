#include "cuda_provider.h"

#include <iostream>
#include <stdexcept>

#include "cuda_memory_pool.h"
#include "device.h"
#include "kernels.h"
#include "onnx_helper.h"
#include "operators.h"
#include "optypes.h"

template <typename T>
void validate_gemm_inputs(const Tensor<T> &A, const Tensor<T> &B,
                          const Tensor<T> &bias, bool transA, bool transB);

CudaProvider::CudaProvider() : allocator_(std::make_shared<CudaAllocator>()) {}

Tensor<float> CudaProvider::evaluateNode(
    const Node *node, const std::vector<Tensor<float> *> &inputs) {
    for (auto input : inputs) {
        input->to(DeviceType::CUDA, allocator_);
    }

    switch (node->getOpType()) {
        case OpType::Gemm:
            return gemm(node, inputs);
        case OpType::Flatten:
            return flatten(node, inputs);
        case OpType::Relu:
            return relu(node, inputs);
        case OpType::Add:
            return add(node, inputs);
        default:
            throw std::runtime_error("Unsupported operation type");
    }
}

Tensor<float> CudaProvider::gemm(const Node *node,
                                 const std::vector<Tensor<float> *> &inputs) {
    if (inputs.size() != 3) {
        throw std::runtime_error("Gemm operation expects 3 inputs");
    }

    float alpha = node->getAttribute<float>("alpha").value_or(1.0);
    float beta = node->getAttribute<float>("beta").value_or(1.0);
    int transA = node->getAttribute<int64_t>("transA").value_or(0);
    int transB = node->getAttribute<int64_t>("transB").value_or(0);

    const Tensor<float> &A = *inputs[0];
    const Tensor<float> &B = *inputs[1];
    const Tensor<float> &bias = *inputs[2];

    validate_gemm_inputs(A, B, bias, transA, transB);

    // Calculate output dimensions depending on transpositions.
    uint64_t N = transA ? A.shape()[1] : A.shape()[0];
    uint64_t M = transB ? B.shape()[1] : B.shape()[0];
    uint64_t K = transB ? B.shape()[0] : B.shape()[1];

    std::vector<uint64_t> dims{N, K};

    Tensor<float> out{std::move(dims), allocator_};

    assert(A.device() == DeviceType::CUDA);
    const float *AData = A.data();
    const float *BData = B.data();
    const float *BiasData = bias.data();

    gemm_cuda(AData, BData, BiasData, out.data(), N, M, K, transA, transB,
              alpha, beta);

    return out;
}

Tensor<float> CudaProvider::flatten(
    const Node *node, const std::vector<Tensor<float> *> &inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("Flatten operation expects 1 input");
    }
    auto axisOpt = node->getAttribute<int64_t>("axis");
    if (!axisOpt) {
        throw std::runtime_error("Axis missing for flatten operation");
    }
    return CudaOperators<float>::flatten(*inputs[0], axisOpt.value());
}

Tensor<float> CudaProvider::relu([[maybe_unused]] const Node *node,
                                 const std::vector<Tensor<float> *> &inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("Relu operation expects 1 input");
    }

    const Tensor<float> &in = *inputs[0];
    Tensor<float> out(in.shape(), allocator_);
    relu_cuda(in.data(), out.data(), out.size());
    return out;
}

Tensor<float> CudaProvider::add([[maybe_unused]] const Node *node,
                                const std::vector<Tensor<float> *> &inputs) {
    if (inputs.size() != 2) {
        throw std::runtime_error("Add operation expects 2 inputs");
    }
    const Tensor<float> &A = *inputs[0];
    const Tensor<float> &B = *inputs[1];
    Tensor<float> out(A.shape(), allocator_);
    add_cuda(A.data(), B.data(), out.data(), out.size());
    return out;
}
