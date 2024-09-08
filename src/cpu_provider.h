#ifndef CPU_PROVIDER_H
#define CPU_PROVIDER_H

#include <memory>
#include <string>
#include <unordered_map>

#include "execution_provider.h"
#include "graph.h"
#include "tensor.h"

class CpuProvider : public ExecutionProvider {
   public:
    Tensor<float> evaluateNode(
        const Node *node, const std::vector<Tensor<float> *> &inputs) override;
    void transferWeightsToDevice(
        [[maybe_unused]] std::unordered_map<std::string, Tensor<float>>
            &weights) override {
        // No-op for CPU provider.
        return;
    }

   private:
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

#endif  // CPU_PROVIDER_H