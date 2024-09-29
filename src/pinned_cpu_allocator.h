#ifndef PINNED_CPU_ALLOCATOR_H
#define PINNED_CPU_ALLOCATOR_H

#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>

#include "memory_allocator.h"
class PinnedMemoryPool {
   public:
    PinnedMemoryPool(size_t size) : total_size_(size) {
        // Use cudaMallocHost instead of cudaMalloc
        cudaMallocHost(&pool_, total_size_);
        free_blocks_.push_back({0, total_size_});
    }

    ~PinnedMemoryPool() {
        // Use cudaFreeHost instead of cudaFree
        cudaFreeHost(pool_);
    }

    void* allocate(size_t size) {
        auto it = std::find_if(
            free_blocks_.begin(), free_blocks_.end(),
            [size](const auto& block) { return block.second >= size; });

        if (it == free_blocks_.end()) {
            throw std::runtime_error("Pinned memory pool exhausted");
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

class PinnedCpuAllocator : public Allocator {
   public:
    PinnedCpuAllocator(size_t pool_size = 100 * 1024 * 1024)  // 100 MiB
        : pool_(std::make_unique<PinnedMemoryPool>(pool_size)) {}

    void* allocate(size_t size) override {
        if (pool_) {
            return pool_->allocate(size);
        }
        void* ptr = nullptr;
        cudaMallocHost(&ptr, size);  // Allocate pinned memory
        return ptr;
    }

    void deallocate(void* ptr) override {
        if (pool_) {
            return pool_->deallocate(ptr);
        }
        cudaFreeHost(&ptr);
    }

    DeviceType getDeviceType() const override { return DeviceType::CPU; }

   private:
    std::unique_ptr<PinnedMemoryPool> pool_;
};

#endif