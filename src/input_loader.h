#ifndef INPUT_LOADER_H
#define INPUT_LOADER_H

#include <string>
#include "tensor.h" // Assuming you have a Tensor class defined

Tensor<float> load_input(const std::string& filename); 

#endif // INPUT_LOADER_H