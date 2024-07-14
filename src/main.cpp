#include <iostream>
#include <assert.h>

#include "input_loader.h"
#include "tensor.h"
#include "model_loader.h"
#include "inference_engine.h"

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <input.data>" << std::endl;
        return 1;
    }

    std::string modelFile = argv[1];
    std::string inputFile = argv[2];

    ModelLoader loader;
    std::unique_ptr<InferenceEngine> engine = loader.load(modelFile);

    Tensor<float> input = load_input(inputFile);

    Tensor<float> output = engine->infer(input);

    return 0;
}
