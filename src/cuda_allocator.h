#ifndef CUDA_ALLOCATOR_H
#define CUDA_ALLOCATOR_H

#include <cuda_runtime.h>

#include <cstring>
#include <memory>

#include "memory_allocator.h"

class CudaAllocator : public Allocator {
   public:
    void* allocate(size_t size) override {
        void* ptr;
        cudaMalloc(&ptr, size);
        return ptr;
    }
    void deallocate(void* ptr) override { cudaFree(ptr); }
    DeviceType getDeviceType() const override { return DeviceType::CUDA; }
};
#endif