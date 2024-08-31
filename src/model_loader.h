#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <memory>
#include <string>
#include <unordered_map>

#include "inference_engine.h"
#include "model_config.h"

class ModelLoader {
   public:
    std::unique_ptr<InferenceEngine> load(const ModelConfig& config);

   private:
    void validate_model(const onnx::ModelProto& model,
                        const ModelConfig& config);
    std::unordered_map<std::string, Tensor<float>> load_weights(
        const onnx::ModelProto& model, const ModelConfig& config);
};

#endif  // MODEL_LOADER_H