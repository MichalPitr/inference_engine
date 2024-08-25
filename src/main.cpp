#include <iostream>
#include <sstream>

#include "input_loader.h"
#include "model_config.h"
#include "model_loader.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml> <input.ubyte>"
                  << std::endl;
        return 1;
    }

    ModelConfig config(argv[1]);
    ModelLoader loader;
    std::cout << "Model path: " << config.get_model_path() << std::endl;
    std::cout << "Execution provider: "
              << (config.get_execution_provider() == ExecutionProvider::CPU
                      ? "CPU"
                      : "CUDA")
              << std::endl;
    std::cout << "Batch size: " << config.get_batch_size() << std::endl;

    auto engine = loader.load(config);
    auto input = load_input(argv[2]);
    auto output = engine->infer(input);
    std::cout << "Out: " << output.to_string() << "\n";

    return 0;
}