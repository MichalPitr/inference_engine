enable_testing()

# Unit tests
foreach(test_name gemm tensor node operators)
  add_executable(${test_name}_test ${test_name}_test.cpp) 
  target_link_libraries(${test_name}_test PRIVATE engine_lib GTest::GTest GTest::Main)
  add_test(NAME ${test_name}_test COMMAND ${test_name}_test)
endforeach()

# Benchmakrs
add_executable(infer_bench infer_bench.cpp)
target_link_libraries(infer_bench PRIVATE engine_lib benchmark::benchmark)