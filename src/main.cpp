#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>

#include "cpu_provider.h"
#include "cuda_provider.h"
#include "cuda_provider_unoptimized.h"
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
    } else if (device == Device::CUDA_SLOW) {
        provider = std::make_unique<CudaProviderUnoptimized>();
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

    // Preload all inputs into memory
    int loops{1000};
    int inferences{100};
    int total_inferences{loops * inferences};
    std::vector<Tensor<float>> inputs;
    inputs.reserve(total_inferences);
    for (int j = 0; j < loops; ++j) {
        for (int i = 0; i < inferences; ++i) {
            std::ostringstream oss;
            oss << file << i << ".ubyte";
            std::string formattedString = oss.str();
            inputs.push_back(load_input(formattedString, config));
        }
    }

    // Create mini batches. Batch size is configured via yaml file.
    std::vector<Tensor<float>> mini_batches;
    int i = 0;
    while (i < total_inferences) {
        std::vector<Tensor<float>> batch;
        for (int j = 0; j < config.get_batch_size() && i + j < total_inferences;
             ++j) {
            batch.push_back(inputs[i + j]);
        }
        if (batch.size() > 0) {
            mini_batches.push_back(Tensor<float>::stack(batch));
        }
        i += batch.size();
    }

    // Inference
    std::vector<Tensor<float>> res;
    res.reserve(total_inferences);
    start = std::chrono::high_resolution_clock::now();
    for (auto batch : mini_batches) {
        session.set_input(config.get_inputs()[0].name, std::move(batch));
        session.run();
        res.push_back(session.get_output(config.get_outputs()[0].name));
    }
    end = std::chrono::high_resolution_clock::now();

    for (auto v : res) {
        std::cout << "Out: " << v.toString() << "\n";
    }

    duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Loop took: " << duration.count() << " microseconds"
              << std::endl;
    std::cout << "Avg inference duration: "
              << duration.count() / total_inferences << " microseconds"
              << std::endl;

    auto duration_s =
        std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    std::cout << "throughput = " << 1000 * total_inferences / duration_s.count()
              << "\n";

    return 0;
}
