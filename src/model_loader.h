#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <memory>
#include <string>

#include "inference_engine.h"

class ModelLoader {
public:
    std::unique_ptr<InferenceEngine> load(const std::string& modelFile);
};

#endif // MODEL_LOADER_H
