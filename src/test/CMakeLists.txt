enable_testing()

# Unit tests
foreach(test_name gemm tensor node operators)
  add_executable(${test_name}_test ${test_name}_test.cpp) 
  target_link_libraries(${test_name}_test PRIVATE engine_lib GTest::GTest GTest::Main)
  add_test(NAME ${test_name}_test COMMAND ${test_name}_test)
endforeach()

# Benchmakrs

# GEMM Benchmark
add_executable(gemm_bench gemm_bench.cpp)
target_link_libraries(gemm_bench 
    PRIVATE 
    engine_lib 
    benchmark::benchmark 
    ${CUDA_LIBRARIES}
)
target_include_directories(gemm_bench PRIVATE ${CUDA_INCLUDE_DIRS})

# Set CUDA properties for gemm_bench
set_target_properties(gemm_bench PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)
