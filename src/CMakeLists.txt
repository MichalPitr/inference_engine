
# Generate the .h and .cxx files
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS onnx-ml.proto)

# Print path to generated files
message ("PROTO_SRCS = ${PROTO_SRCS}")
message ("PROTO_HDRS = ${PROTO_HDRS}")

# Add an executable
add_executable(${PROJECT_NAME}
    main.cpp
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