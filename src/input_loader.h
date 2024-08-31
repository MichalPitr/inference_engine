#ifndef INPUT_LOADER_H
#define INPUT_LOADER_H

#include <string>

#include "model_config.h"
#include "tensor.h"

Tensor<float> load_input(const std::string& filename,
                         const ModelConfig& config);

#endif  // INPUT_LOADER_H