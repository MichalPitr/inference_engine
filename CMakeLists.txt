cmake_minimum_required(VERSION 3.22)
project(engine)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-deprecated-declarations -Wno-unused-function")

# # Flags for profiling
# set(CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_PROFILE} -g -O3 -fno-omit-frame-pointer")
# set(CMAKE_EXE_LINKER_FLAGS_PROFILE "${CMAKE_EXE_LINKER_FLAGS_PROFILE} -g") 

# set(CMAKE_BUILD_TYPE Profile)

# Find Protobuf
find_package(Protobuf REQUIRED)
find_package(benchmark REQUIRED)
find_package(GTest REQUIRED)

include_directories(${PROTOBUF_INCLUDE_DIRS})


# Add subdirectories
add_subdirectory(src)
add_subdirectory(src/test)
