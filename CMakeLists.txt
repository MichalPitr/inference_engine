cmake_minimum_required(VERSION 3.10)
project(MyInferenceEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the source files
set(SOURCES src/main.cpp)

# Create the executable
add_executable(inference_engine ${SOURCES})
