#ifndef MEMORY_ALLOCATOR_H
#define MEMORY_ALLOCATOR_H

#include <cstring>

#include "device.h"

class Allocator {
   public:
    virtual void* allocate(size_t size) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual DeviceType getDeviceType() const = 0;
    virtual ~Allocator() = default;
};

#endif