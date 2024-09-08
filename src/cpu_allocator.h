#ifndef CPU_ALLOCATOR_H
#define CPU_ALLOCATOR_H

#include <cstring>
#include <memory>

#include "memory_allocator.h"

class CpuAllocator : public Allocator {
   public:
    void* allocate(size_t size) override { return malloc(size); }

    void deallocate(void* ptr) override { free(ptr); }
    DeviceType getDeviceType() const override { return DeviceType::CPU; };
};

#endif