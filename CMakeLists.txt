cmake_minimum_required(VERSION 3.22)
project(engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-deprecated-declarations -Wno-unused-function")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra,-Wno-deprecated-declarations,-Wno-unused-function")

find_package(Protobuf REQUIRED)
find_package(benchmark REQUIRED)
find_package(GTest REQUIRED)
find_package(yaml-cpp REQUIRED)

include(FindCUDAToolkit)
if(CMAKE_CUDA_COMPILER)
    add_definitions(-DUSE_CUDA)
endif()

include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${YAML_CPP_INCLUDE_DIR})

add_subdirectory(src)
add_subdirectory(src/test)
