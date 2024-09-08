#ifndef CUDA_PROVIDER_H
#define CUDA_PROVIDER_H

#include <memory>
#include <string>
#include <unordered_map>

#include "cuda_memory_pool.h"
#include "execution_provider.h"
#include "graph.h"
#include "tensor.h"

class CudaProvider : public ExecutionProvider {
   public:
    CudaProvider();
    ~CudaProvider() override = default;

    Tensor<float> evaluateNode(
        const Node *node, const std::vector<Tensor<float> *> &inputs) override;

   private:
    std::unique_ptr<CudaMemoryPool> memory_pool_;

    // Operators
    Tensor<float> gemm(const Node *node,
                       const std::vector<Tensor<float> *> &inputs);
    Tensor<float> flatten(const Node *node,
                          const std::vector<Tensor<float> *> &inputs);
    Tensor<float> relu(const Node *node,
                       const std::vector<Tensor<float> *> &inputs);
    Tensor<float> add(const Node *node,
                      const std::vector<Tensor<float> *> &inputs);
};

#endif  // CUDA_PROVIDER_H