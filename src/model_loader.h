#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <memory>
#include <string>

#include "inference_engine.h"

class ModelLoader {
   public:
    std::unique_ptr<InferenceEngine> load(const std::string &modelFile);

   private:
    void validate_model(const onnx::ModelProto &model);
    std::unordered_map<std::string, Tensor<float>> load_weights(
        const onnx::ModelProto &model);
};

#endif  // MODEL_LOADER_H
