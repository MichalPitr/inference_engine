
# Generate the .h and .cxx files
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS onnx-ml.proto)

# Force shared library
set(BUILD_SHARED_LIBS ON)

# Add an executable
add_library(engine_lib
    gemm_cpu.cpp
    kernels.cu
    attribute.cpp
    tensor.cpp
    input_loader.cpp
    operators.cpp
    graph.cpp
    node.cpp
    onnx_helper.cpp
    model_config.cpp
    inference_session.cpp
    cpu_provider.cpp
    cuda_provider.cpp
    cuda_provider_unoptimized.cpp
    ${PROTO_SRCS}
    ${PROTO_HDRS})

target_include_directories(engine_lib PUBLIC
    ${PROTOBUF_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${YAML_CPP_INCLUDE_DIR})

# link the exe against the libraries
target_link_libraries(engine_lib PUBLIC ${PROTOBUF_LIBRARIES}  yaml-cpp)

add_executable(engine_exe main.cpp)

target_link_libraries(engine_exe PRIVATE engine_lib)
