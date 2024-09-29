#ifndef INFERENCE_WRAPPER_H
#define INFERENCE_WRAPPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct InferenceSessionWrapper InferenceSessionWrapper;

typedef struct {
    float* data;
    int size;
} InferenceResult;

InferenceSessionWrapper* create_session(const char* config_path);
void destroy_session(InferenceSessionWrapper* session);
int initialize_provider(InferenceSessionWrapper* session);
InferenceResult run_inference(InferenceSessionWrapper* session,
                              float* input_data, uint64_t input_size);
void free_result(InferenceResult result);

#ifdef __cplusplus
}
#endif

#endif  // INFERENCE_WRAPPER_H