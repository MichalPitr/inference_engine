#include <iostream>
#include <sstream>

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
    
    // 100 Sequential inference requests. 
    std::string file = "/home/michal/code/inference_engine/inputs/image_";
    // for (int i = 0; i < 1; ++i) {
        // std::ostringstream oss;
        // oss << file << i << ".ubyte";
        // std::string formattedString = oss.str();
    Tensor<float> input = load_input(inputFile);
    engine->applyOptimizations();
    Tensor<float> output = engine->infer(input);
    std::cout << "Out: " << output.to_string() << "\n";
    // }

    return 0;
}
