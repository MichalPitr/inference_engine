#ifndef CUDA_MEMORY_POOL_H
#define CUDA_MEMORY_POOL_H

#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

class CudaMemoryPool {
   public:
    CudaMemoryPool(size_t size) : total_size_(size), used_size_(0) {
        cudaMalloc(&pool_, total_size_);
    }

    ~CudaMemoryPool() { cudaFree(pool_); }

    void* allocate(size_t size) {
        if (used_size_ + size > total_size_) {
            throw std::runtime_error("Memory pool exhausted");
        }
        void* ptr = static_cast<char*>(pool_) + used_size_;
        used_size_ += size;
        allocations_[ptr] = size;
        return ptr;
    }

    void deallocate(void* ptr) {
        auto it = allocations_.find(ptr);
        if (it == allocations_.end()) {
            throw std::runtime_error(
                "Attempt to deallocate unallocated memory");
        }
        allocations_.erase(it);
        // Note: We don't reduce used_size_ here to keep it simple
        // In a more advanced implementation, we might want to manage free
        // blocks
    }

    void reset() {
        used_size_ = 0;
        allocations_.clear();
    }

   private:
    void* pool_;
    size_t total_size_;
    size_t used_size_;
    std::unordered_map<void*, size_t> allocations_;
};
#endif
