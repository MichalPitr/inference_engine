#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

#include "cuda_memory_pool.h"

CudaMemoryPool::CudaMemoryPool() {}

CudaMemoryPool::~CudaMemoryPool() {
    for (auto& block : memory_blocks_) {
        cudaFree(block.ptr);
    }
}

void* CudaMemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Try to find a suitable block
    auto it = std::find_if(memory_blocks_.begin(), memory_blocks_.end(),
                           [size](const MemoryBlock& block) {
                               return !block.in_use && block.size >= size;
                           });

    if (it != memory_blocks_.end()) {
        it->in_use = true;
        return it->ptr;
    }

    // If no suitable block found, allocate a new one
    return allocateNew(size);
}

void CudaMemoryPool::deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = std::find_if(
        memory_blocks_.begin(), memory_blocks_.end(),
        [ptr](const MemoryBlock& block) { return block.ptr == ptr; });

    if (it != memory_blocks_.end()) {
        it->in_use = false;
    }
}

void* CudaMemoryPool::allocateNew(size_t size) {
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memory allocation failed");
    }

    memory_blocks_.push_back({ptr, size, true});
    return ptr;
}