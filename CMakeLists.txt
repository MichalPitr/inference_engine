cmake_minimum_required(VERSION 3.10)
project(engine)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# Enable compiler warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Clang-specific flags or configuration
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
else()
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated-declarations -Wno-unused-function")
endif()

# Find Protobuf
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

# Add subdirectories
add_subdirectory(src)
