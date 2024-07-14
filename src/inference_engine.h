#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <string>

#include "onnx-ml.pb.h"
#include "tensor.h"
#include "operators.h"

class InferenceEngine
{
public:
    InferenceEngine(const onnx::GraphProto &graph,
                    std::unordered_map<std::string, Tensor<float>> weights);
    Tensor<float> infer(const Tensor<float> &input);

private:
    onnx::GraphProto graph_;
    std::unordered_map<std::string, Tensor<float>> weights_;
};

#endif // INFERENCE_ENGINE_H