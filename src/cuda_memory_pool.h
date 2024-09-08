#ifndef CUDA_MEMORY_POOL_H
#define CUDA_MEMORY_POOL_H

#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

class CudaMemoryPool {
   public:
    CudaMemoryPool(size_t size) : total_size_(size) {
        cudaMalloc(&pool_, total_size_);
        free_blocks_.push_back({0, total_size_});
    }

    ~CudaMemoryPool() { cudaFree(pool_); }

    void* allocate(size_t size) {
        auto it = std::find_if(
            free_blocks_.begin(), free_blocks_.end(),
            [size](const auto& block) { return block.second >= size; });

        if (it == free_blocks_.end()) {
            throw std::runtime_error("Memory pool exhausted");
        }

        size_t offset = it->first;
        size_t block_size = it->second;

        if (block_size > size) {
            // Split the block
            it->first += size;
            it->second -= size;
        } else {
            // Use the entire block
            free_blocks_.erase(it);
        }

        return static_cast<char*>(pool_) + offset;
    }

    void deallocate(void* ptr) {
        size_t offset = static_cast<char*>(ptr) - static_cast<char*>(pool_);

        auto it = std::lower_bound(
            free_blocks_.begin(), free_blocks_.end(), offset,
            [](const auto& block, size_t off) { return block.first < off; });

        size_t size = (it != free_blocks_.end()) ? it->first - offset
                                                 : total_size_ - offset;

        if (it != free_blocks_.begin() &&
            (it - 1)->first + (it - 1)->second == offset) {
            // Merge with previous block
            auto prev = it - 1;
            prev->second += size;
            if (it != free_blocks_.end() && offset + size == it->first) {
                // Merge with next block too
                prev->second += it->second;
                free_blocks_.erase(it);
            }
        } else if (it != free_blocks_.end() && offset + size == it->first) {
            // Merge with next block
            it->first = offset;
            it->second += size;
        } else {
            // Insert new block
            free_blocks_.insert(it, {offset, size});
        }
    }

    void reset() {
        free_blocks_.clear();
        free_blocks_.push_back({0, total_size_});
    }

   private:
    void* pool_;
    size_t total_size_;
    std::vector<std::pair<size_t, size_t>> free_blocks_;  // offset, size
};

#endif
