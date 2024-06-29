#include "gtest/gtest.h"
#include "../tensor.h"
#include "../operators.h"

TEST(OperatorsTest, flatten) {
    Tensor<float> t1 {{1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 2, 4}};
    
    Tensor<float> t2 = flatten(t1, uint64_t{0});
    std::vector<uint64_t> expectedShape1{1, 8};
    EXPECT_EQ(expectedShape1, t2.shape());

    Tensor<float> t3 = flatten(t1, uint64_t{1});
    std::vector<uint64_t> expectedShape2{1, 8};
    EXPECT_EQ(expectedShape2, t3.shape());

    Tensor<float> t4 = flatten(t1, uint64_t{2});
    std::vector<uint64_t> expectedShape3{1, 8};
    EXPECT_EQ(expectedShape3, t4.shape());

    Tensor<float> t5 = flatten(t1, uint64_t{3});
    std::vector<uint64_t> expectedShape4{2, 4};
    EXPECT_EQ(expectedShape4, t5.shape());
}

TEST(OperatorsTest, relu) {
    Tensor<float> t1 {{-1., 0, 1, -1, 1, -0.5, -0, 0.5}, {2, 4}};
    auto t2 = relu(t1);
    std::vector<float> expected{0, 0, 1, 0, 1, 0, 0, 0.5};
    EXPECT_EQ(expected, t2.data());

    Tensor<float> t3 {{1, 1, 1, 1}, {2, 2}};
    auto t4 = relu(t3);
    std::vector<float> expected2{1, 1, 1, 1};
    EXPECT_EQ(expected2, t4.data());
}