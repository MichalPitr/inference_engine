#ifndef CUDA_ALLOCATOR_H
#define CUDA_ALLOCATOR_H

#include <cuda_runtime.h>

#include <cstring>
#include <memory>

#include "cuda_memory_pool.h"
#include "memory_allocator.h"

class CudaAllocator : public Allocator {
   public:
    CudaAllocator(size_t pool_size = 200 * 1024 * 1024) {}
    // : pool_(std::make_unique<CudaMemoryPool>(pool_size)) {}

    void* allocate(size_t size) override {
        if (pool_) {
            return pool_->allocate(size);
        }
        // fallback if no pool configured.
        void* ptr = nullptr;
        cudaMalloc(&ptr, size);
        return ptr;
    }
    void deallocate(void* ptr) override {
        if (pool_) {
            return pool_->deallocate(ptr);
        }
        // fallback if no pool configured.
        cudaFree(ptr);
    }
    DeviceType getDeviceType() const override { return DeviceType::CUDA; }

   private:
    std::unique_ptr<CudaMemoryPool> pool_;
};

#endif