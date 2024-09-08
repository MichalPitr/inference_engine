#ifndef CUDA_MEMORY_POOL_H
#define CUDA_MEMORY_POOL_H

#include <cstddef>
#include <mutex>
#include <vector>

class CudaMemoryPool {
   public:
    CudaMemoryPool();
    ~CudaMemoryPool();

    void* allocate(size_t size);
    void deallocate(void* ptr);

    // Disable copy constructor and assignment operator
    CudaMemoryPool(const CudaMemoryPool&) = delete;
    CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

   private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
    };

    std::vector<MemoryBlock> memory_blocks_;
    std::mutex mutex_;

    void* allocateNew(size_t size);
};

#endif
