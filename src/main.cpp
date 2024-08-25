#include <iostream>
#include <sstream>

#include "input_loader.h"
#include "model_config.h"
#include "model_loader.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml>" << std::endl;
        return 1;
    }

    // TODO: implement a simple http/grpc server that can receive inputs in
    // byte64 format, do some naive online batching, and passes the mini batch
    // to the inference engine.
    // This will make usage a lot nicer and more practical. Disadvantage is
    // complexities of simulating load on the server.
    std::string inputFile =
        "/home/michal/code/inference_engine/inputs/image_0.ubyte";

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
    auto input = load_input(inputFile);
    auto output = engine->infer(input);
    std::cout << "Out: " << output.to_string() << "\n";

    return 0;
}