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
    InferenceEngine(std::unique_ptr<Graph> graph,
                    std::unordered_map<std::string, Tensor<float>> weights,
                    DeviceType device);
    Tensor<float> infer(const Tensor<float>& input);
    void applyOptimizations();

   private:
    void applyConstantFolding();
    void registerCpuOperators();
    void registerCudaOperators();

    Tensor<float> evaluateNode(const Node* node,
                               const std::vector<Tensor<float>*>& inputs);
    std::vector<Tensor<float>*> prepareNodeInputs(const Node* node);

    OperatorRegistry<float> registry_;
    std::unique_ptr<Graph> graph_;
    std::unordered_map<std::string, Tensor<float>> weights_;
    DeviceType device_;
};

#endif  // INFERENCE_ENGINE_H