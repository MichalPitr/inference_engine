#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>

#include "cpu_provider.h"
#include "cuda_provider.h"
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
    std::cout << "Device: "
              << (config.get_device() == Device::CPU ? "CPU" : "CUDA")
              << std::endl;
    std::cout << "Batch size: " << config.get_batch_size() << std::endl;

    InferenceSession session;
    session.load_model(config);

    std::unique_ptr<ExecutionProvider> provider;
    Device device = config.get_device();
    if (device == Device::CPU) {
        provider = std::make_unique<CpuProvider>();
    } else if (device == Device::CUDA) {
        provider = std::make_unique<CudaProvider>();
    } else {
        throw std::runtime_error("Unknown device type");
    }

    session.set_execution_provider(std::move(provider));

    // Moves model weights to device memory.
    auto start = std::chrono::high_resolution_clock::now();
    session.initialize_provider();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "initialize_provider took: " << duration.count()
              << " microseconds" << std::endl;

    std::string file = "/home/michal/code/inference_engine/inputs/image_";

    int loops{1};
    int inferences{100};
    std::vector<Tensor<float>> inputs;
    for (int j = 0; j < loops; ++j) {
        for (int i = 0; i < inferences; ++i) {
            std::ostringstream oss;
            oss << file << i << ".ubyte";
            std::string formattedString = oss.str();
            auto input = load_input(formattedString, config);
            inputs.push_back(input);
        }
    }

    start = std::chrono::high_resolution_clock::now();
    for (auto input : inputs) {
        session.set_input("onnx::Flatten_0", input);

        session.run();

        auto output = session.get_output("21");

        std::cout << "Out: " << output.toString() << "\n";
    }

    int total_inferences{loops * inferences};
    end = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Loop took: " << duration.count() << " microseconds"
              << std::endl;
    std::cout << "Avg inference duration: "
              << duration.count() / total_inferences << " microseconds"
              << std::endl;

    return 0;
}
