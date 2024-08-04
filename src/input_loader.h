#ifndef INPUT_LOADER_H
#define INPUT_LOADER_H

#include <string>
#include "tensor.h"

Tensor<float> load_input(const std::string& filename); 

#endif // INPUT_LOADER_H