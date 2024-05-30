cmake_minimum_required(VERSION 3.10)
project(engine)

set(CMAKE_BUILD_TYPE Debug)

# Find Protobuf
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

# Add subdirectories
add_subdirectory(src)
