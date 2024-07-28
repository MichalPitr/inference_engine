#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <string>

#include "tensor.h"
#include "graph.h"

class InferenceEngine
{
public:
    InferenceEngine(std::unique_ptr<Graph> graph,
                    std::unordered_map<std::string, Tensor<float>> weights);
    Tensor<float> infer(const Tensor<float> &input);

private:
    std::unique_ptr<Graph> graph_;
    std::unordered_map<std::string, Tensor<float>> weights_;
};

#endif // INFERENCE_ENGINE_H