
# Generate the .h and .cxx files
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS onnx-ml.proto)

# Print path to generated files
message ("PROTO_SRCS = ${PROTO_SRCS}")
message ("PROTO_HDRS = ${PROTO_HDRS}")


# Add an executable
add_executable(${PROJECT_NAME}
    main.cpp
    gemm.cpp
    tensor.cpp
    ${PROTO_SRCS}
    ${PROTO_HDRS})

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${PROTOBUF_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

# link the exe against the libraries
target_link_libraries(${PROJECT_NAME}
    PUBLIC
    ${PROTOBUF_LIBRARIES}
)

find_package(GTest REQUIRED)
enable_testing()

add_executable(gemm_test test/gemm_test.cpp gemm.cpp)
target_link_libraries(gemm_test GTest::GTest GTest::Main)
add_test(NAME gemm_test COMMAND gemm_test)

add_executable(tensor_test test/tensor_test.cpp tensor.cpp)
target_link_libraries(tensor_test GTest::GTest GTest::Main)
add_test(NAME tensor_test COMMAND tensor_test)