#include <benchmark/benchmark.h>

#include <memory>

#include "../inference_engine.h"
#include "../input_loader.h"
#include "../model_loader.h"
#include "../tensor.h"

// Global setup: Load the model and input ONCE
std::unique_ptr<InferenceEngine> global_engine;
Tensor<float> global_input;

static void BM_Infer(benchmark::State& state) {
    // Perform inference for each iteration
    for (auto _ : state) {
        benchmark::DoNotOptimize(global_engine->infer(global_input));
    }
}

// Register the benchmark function
BENCHMARK(BM_Infer);

int main(int argc, char** argv) {
    // Load model and input before running benchmarks
    ModelLoader loader;
    global_engine =
        loader.load("/home/michal/code/inference_engine/models/mnist_ffn.onnx");
    global_input =
        load_input("/home/michal/code/inference_engine/inputs/image_0.ubyte");

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}