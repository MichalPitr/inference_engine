
# Generate the .h and .cxx files
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS onnx-ml.proto)

# Add an executable
add_library(engine_lib
    main.cpp
    gemm.cpp
    tensor.cpp
    input_loader.cpp
    operators.cpp
    graph.cpp
    node.cpp
    onnx_helper.cpp
    model_loader.cpp
    inference_engine.cpp
    ${PROTO_SRCS}
    ${PROTO_HDRS})

target_include_directories(engine_lib PUBLIC
    ${PROTOBUF_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR})

# link the exe against the libraries
target_link_libraries(engine_lib PUBLIC ${PROTOBUF_LIBRARIES})

add_executable(engine_exe main.cpp)
target_link_libraries(engine_exe PRIVATE engine_lib)
