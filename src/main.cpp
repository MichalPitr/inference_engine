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
    std::string file = "/home/michal/code/inference_engine/inputs/image_";
    // Rerun 100 times.
    for (int j = 0; j < 5; ++j) {
        // 100 Sequential inference requests.
        for (int i = 0; i < 100; ++i) {
            std::ostringstream oss;
            oss << file << i << ".ubyte";
            std::string formattedString = oss.str();
            auto input = load_input(formattedString, config);
            // auto input = load_input(argv[2]);

            auto output = engine->infer(input);
            std::cout << "Out: " << output.toString() << "\n";
        }
    }

    return 0;
}