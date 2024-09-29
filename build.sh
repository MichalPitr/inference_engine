#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Define directories using absolute paths
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
BUILD_DIR="${PROJECT_ROOT}/build"
SERVER_DIR="${PROJECT_ROOT}/server"
GENERATED_HEADERS_DIR="${BUILD_DIR}/src"

echo "Project root: ${PROJECT_ROOT}"

# Step 1: Create build directory if it doesn't exist
mkdir -p "${BUILD_DIR}"

# Step 2: Run CMake
/usr/bin/cmake --build "${BUILD_DIR}" --config Release --target engine_exe -j 14 --

# Step 3: Create a symbolic link for the shared library
ln -sf "${GENERATED_HEADERS_DIR}/libengine_lib.so" "${SERVER_DIR}/libengine_lib.so"
echo "Created symlink for libengine_lib.so"

# Step 4: Create a symbolic link for the onnx-ml.pb.h header
ln -sf "${GENERATED_HEADERS_DIR}/onnx-ml.pb.h" "${SERVER_DIR}/onnx-ml.pb.h"
echo "Created symlink for onnx-ml.pb.h"

# Step 5: Build the Go server
cd "${SERVER_DIR}"
go build

echo "Build process completed successfully!"