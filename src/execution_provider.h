#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <memory>
#include <string>
#include <unordered_map>

#include "graph.h"
#include "operator_registry.h"
#include "tensor.h"

class ExecutionProvider {
   public:
    ExecutionProvider(DeviceType device);
    Tensor<float> evaluateNode(const Node* node,
                               const std::vector<Tensor<float>*>& inputs);

   private:
    void registerCpuOperators();
    void registerCudaOperators();

    OperatorRegistry<float> registry_;
    DeviceType device_;
};

#endif  // INFERENCE_ENGINE_H