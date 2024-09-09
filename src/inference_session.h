#ifndef INFERENCE_SESSION_H
#define INFERENCE_SESSION_H

#include <memory>
#include <string>
#include <unordered_map>

#include "execution_provider.h"
#include "graph.h"
#include "model_config.h"
#include "pinned_cpu_allocator.h"
#include "tensor.h"

class InferenceSession {
   public:
    void load_model(const ModelConfig& config);
    void set_execution_provider(std::unique_ptr<ExecutionProvider> engine);
    void initialize_provider();
    void set_input(const std::string& name, Tensor<float> input);
    Tensor<float> get_output(const std::string& name);
    void run();

   private:
    std::vector<Tensor<float>*> prepare_node_inputs(const Node* node);

    std::shared_ptr<PinnedCpuAllocator> pinned_allocator_;
    std::unique_ptr<ExecutionProvider> engine_;
    std::unique_ptr<Graph> graph_;
    std::unordered_map<std::string, Tensor<float>> weights_;
};

#endif
