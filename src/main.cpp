#include <iostream>
#include <memory>
#include <sstream>

#include "inference_engine.h"
#include "inference_session.h"
#include "input_loader.h"
#include "model_config.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml> <input.ubyte>"
                  << std::endl;
        return 1;
    }

    ModelConfig config(argv[1]);
    std::cout << "Model path: " << config.get_model_path() << std::endl;
    std::cout << "Execution provider: "
              << (config.get_execution_provider() == ExecutionProvider::CPU
                      ? "CPU"
                      : "CUDA")
              << std::endl;
    std::cout << "Batch size: " << config.get_batch_size() << std::endl;

    InferenceSession inference_session;
    inference_session.load_model(config);
    inference_session.set_execution_provider(
        std::make_unique<InferenceEngine>(DeviceType::CPU));

    std::string file = "/home/michal/code/inference_engine/inputs/image_";
    for (int j = 0; j < 1; ++j) {
        for (int i = 0; i < 1; ++i) {
            std::ostringstream oss;
            oss << file << i << ".ubyte";
            std::string formattedString = oss.str();
            auto input = load_input(formattedString, config);
            inference_session.set_input("onnx::Flatten_0", input);

            inference_session.run();

            auto output = inference_session.get_output("21");

            std::cout << "Out: " << output.toString() << "\n";
        }
    }

    return 0;
}
