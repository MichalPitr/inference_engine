#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <memory>
#include <string>
#include <unordered_map>

#include "graph.h"
#include "tensor.h"

class InferenceEngine {
   public:
    InferenceEngine(std::unique_ptr<Graph> graph,
                    std::unordered_map<std::string, Tensor<float>> weights);
    Tensor<float> infer(const Tensor<float>& input);
    void applyOptimizations();

   private:
    void applyConstantFolding();
    Tensor<float> evaluateNode(const Node* node,
                               const std::vector<Tensor<float>*>& inputs);
    std::vector<Tensor<float>*> ptrPrepareNodeInputs(const Node* node);

    std::unique_ptr<Graph> graph_;
    std::unordered_map<std::string, Tensor<float>> weights_;
};

#endif  // INFERENCE_ENGINE_H