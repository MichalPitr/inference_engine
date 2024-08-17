#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <string>
#include <memory>
#include <unordered_map>

#include "tensor.h"
#include "graph.h"

class InferenceEngine
{
public:
    InferenceEngine(std::unique_ptr<Graph> graph,
                    std::unordered_map<std::string, Tensor<float>> weights);
    Tensor<float> infer(const Tensor<float> &input);
    void applyOptimizations();

private:
    void applyConstantFolding();
    Tensor<float> evaluateNode(const Node *node, std::vector<Tensor<float>*> inputs);
    std::vector<Tensor<float>> prepareNodeInputs(const Node* node);
    std::vector<Tensor<float>*> ptrPrepareNodeInputs(const Node *node);

    std::unique_ptr<Graph> graph_;
    std::unordered_map<std::string, Tensor<float>> weights_;
};

#endif // INFERENCE_ENGINE_H