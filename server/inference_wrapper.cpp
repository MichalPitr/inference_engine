// inference_wrapper.cpp
#include "inference_wrapper.h"

#include <memory>
#include <vector>

#include "cpu_provider.h"
#include "cuda_provider.h"
#include "cuda_provider_unoptimized.h"
#include "inference_session.h"
#include "model_config.h"

struct InferenceSessionWrapper {
    ModelConfig config;
    InferenceSession session;
    std::unique_ptr<ExecutionProvider> provider;
};

extern "C" {

InferenceSessionWrapper* create_session(const char* config_path) {
    auto wrapper = new InferenceSessionWrapper();
    wrapper->config = ModelConfig(config_path);
    wrapper->session.load_model(wrapper->config);

    Device device = wrapper->config.get_device();
    if (device == Device::CPU) {
        wrapper->provider = std::make_unique<CpuProvider>();
    } else if (device == Device::CUDA) {
        wrapper->provider = std::make_unique<CudaProvider>();
    } else if (device == Device::CUDA_SLOW) {
        wrapper->provider = std::make_unique<CudaProviderUnoptimized>();
    } else {
        delete wrapper;
        return nullptr;
    }

    wrapper->session.set_execution_provider(std::move(wrapper->provider));
    return wrapper;
}

void destroy_session(InferenceSessionWrapper* wrapper) { delete wrapper; }

int initialize_provider(InferenceSessionWrapper* wrapper) {
    try {
        wrapper->session.initialize_provider();
        return 0;
    } catch (const std::exception& e) {
        return -1;
    }
}

InferenceResult run_inference(InferenceSessionWrapper* wrapper,
                              float* input_data, uint64_t input_size) {
    Tensor<float> input(input_data, {1, input_size});
    wrapper->session.set_input(wrapper->config.get_inputs()[0].name,
                               std::move(input));
    wrapper->session.run();
    Tensor<float> output =
        wrapper->session.get_output(wrapper->config.get_outputs()[0].name);
    InferenceResult result;
    result.size = output.size();
    result.data = (float*)malloc(result.size * sizeof(float));
    std::memcpy(result.data, output.data(), result.size * sizeof(float));

    return result;
}

void free_result(InferenceResult result) { free(result.data); }
}