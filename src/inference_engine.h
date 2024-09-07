#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <memory>
#include <string>
#include <unordered_map>

#include "graph.h"
#include "operator_registry.h"
#include "tensor.h"

class InferenceEngine {
   public:
    InferenceEngine(DeviceType device);
    Tensor<float> evaluateNode(const Node* node,
                               const std::vector<Tensor<float>*>& inputs);
    std::vector<Tensor<float>*> prepareNodeInputs(const Node* node);

   private:
    void registerCpuOperators();
    void registerCudaOperators();

    OperatorRegistry<float> registry_;
    DeviceType device_;
};

#endif  // INFERENCE_ENGINE_H