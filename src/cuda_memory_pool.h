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
   private:
    void* pool_start;
    size_t pool_size;
    struct Block {
        size_t offset;
        size_t size;
        bool is_free;
        Block(size_t off, size_t sz, bool free)
            : offset(off), size(sz), is_free(free) {}
    };
    std::vector<Block> blocks;

   public:
    CudaMemoryPool(size_t size) : pool_size(size) {
        cudaError_t err = cudaMalloc(&pool_start, size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA memory pool");
        }
        blocks.emplace_back(0, size, true);
    }

    ~CudaMemoryPool() { cudaFree(pool_start); }

    void* allocate(size_t size) {
        auto best_fit = blocks.end();
        size_t smallest_fit = pool_size + 1;

        for (auto it = blocks.begin(); it != blocks.end(); ++it) {
            if (it->is_free && it->size >= size) {
                if (it->size < smallest_fit) {
                    best_fit = it;
                    smallest_fit = it->size;
                }
            }
        }

        if (best_fit == blocks.end()) {
            throw std::runtime_error("Out of memory in CUDA pool");
        }

        size_t alloc_offset = best_fit->offset;
        size_t remaining_size = best_fit->size - size;

        best_fit->is_free = false;
        best_fit->size = size;

        if (remaining_size > 0) {
            blocks.insert(best_fit + 1,
                          Block(alloc_offset + size, remaining_size, true));
        }

        return static_cast<char*>(pool_start) + alloc_offset;
    }

    void deallocate(void* ptr) {
        size_t offset =
            static_cast<char*>(ptr) - static_cast<char*>(pool_start);
        auto it = std::find_if(
            blocks.begin(), blocks.end(),
            [offset](const Block& b) { return b.offset == offset; });

        if (it == blocks.end()) {
            throw std::runtime_error("Invalid pointer for deallocation");
        }

        it->is_free = true;

        // Coalesce with previous block if free
        if (it != blocks.begin()) {
            auto prev = it - 1;
            if (prev->is_free) {
                prev->size += it->size;
                blocks.erase(it);
                it = prev;
            }
        }

        // Coalesce with next block if free
        if (it != blocks.end() - 1) {
            auto next = it + 1;
            if (next->is_free) {
                it->size += next->size;
                blocks.erase(next);
            }
        }
    }
};

#endif