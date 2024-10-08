#include "../operators.h"

#include "../tensor.h"
#include "gtest/gtest.h"

TEST(OperatorsTest, flatten) {
    Tensor<float> t1{{1, 2, 3, 4, 5, 6, 7, 8}, {1, 1, 2, 4}};

    Tensor<float> t2 = CpuOperators<float>::flatten(t1, uint64_t{0});
    std::vector<uint64_t> expectedShape1{1, 8};
    EXPECT_EQ(expectedShape1, t2.shape());

    Tensor<float> t3 = CpuOperators<float>::flatten(t1, uint64_t{1});
    std::vector<uint64_t> expectedShape2{1, 8};
    EXPECT_EQ(expectedShape2, t3.shape());

    Tensor<float> t4 = CpuOperators<float>::flatten(t1, uint64_t{2});
    std::vector<uint64_t> expectedShape3{1, 8};
    EXPECT_EQ(expectedShape3, t4.shape());

    Tensor<float> t5 = CpuOperators<float>::flatten(t1, uint64_t{3});
    std::vector<uint64_t> expectedShape4{2, 4};
    EXPECT_EQ(expectedShape4, t5.shape());
}

TEST(OperatorsTest, relu) {
    Tensor<float> t1{{-1., 0, 1, -1, 1, -0.5, -0, 0.5}, {2, 4}};
    auto t2 = CpuOperators<float>::relu(t1);
    std::vector<float> expected{0, 0, 1, 0, 1, 0, 0, 0.5};

    for (std::size_t i = 0; i < t2.size(); ++i) {
        EXPECT_EQ(t2.data()[i], expected[i]);
    }

    Tensor<float> t3{{1, 1, 1, 1}, {2, 2}};
    auto t4 = CpuOperators<float>::relu(t3);
    std::vector<float> expected2{1, 1, 1, 1};
    for (std::size_t i = 0; i < t4.size(); ++i) {
        EXPECT_EQ(t4.data()[i], expected2[i]);
    }
}

TEST(OperatorsTest, gemmMatrixVector) {
    Tensor<float> A{{1, 2, 3, 4}, {2, 2}};
    Tensor<float> B{{1, 1}, {2, 1}};
    Tensor<float> bias{{1, 1}, {2, 1}};

    auto res = CpuOperators<float>::gemm(A, B, bias, false, false, 1, 1);

    std::vector<uint64_t> expectShape{2, 1};
    EXPECT_EQ(expectShape, res.shape());
    std::vector<float> expectData{4, 8};
    for (std::size_t i = 0; i < res.size(); ++i) {
        EXPECT_EQ(res.data()[i], expectData[i]);
    }
}

TEST(OperatorsTest, gemmMatrixMatrix) {
    Tensor<float> A{{1, 2, 3, 4}, {2, 2}};
    Tensor<float> B{{1, 1, 1, 1}, {2, 2}};
    Tensor<float> bias{{1, 2}, {2, 1}};

    auto res = CpuOperators<float>::gemm(A, B, bias, false, false, 1, 1);

    std::vector<uint64_t> expectShape{2, 2};
    EXPECT_EQ(expectShape, res.shape());

    std::vector<float> expectData{4, 4, 9, 9};
    for (std::size_t i = 0; i < res.size(); ++i) {
        EXPECT_EQ(res.data()[i], expectData[i]);
    }
}

TEST(OperatorsTest, gemmMatrixMatrixTransA) {
    Tensor<float> A{{1, 2}, {2, 1}};
    Tensor<float> B{{1, 1, 1, 1}, {2, 2}};
    Tensor<float> bias{{0, 0}, {2}};

    auto res = CpuOperators<float>::gemm(A, B, bias, true, false, 1, 1);

    std::vector<uint64_t> expectShape{1, 2};
    EXPECT_EQ(expectShape, res.shape());

    std::vector<float> expectData{3, 3};
    for (std::size_t i = 0; i < res.size(); ++i) {
        EXPECT_EQ(res.data()[i], expectData[i]);
    }
}

TEST(OperatorsTest, gemmMatrixMatrixTransB) {
    Tensor<float> A{{1, 1, 1, 1}, {2, 2}};
    Tensor<float> B{{1, 2}, {1, 2}};
    Tensor<float> bias{{0, 0}, {2}};

    auto res = CpuOperators<float>::gemm(A, B, bias, false, true, 1, 1);

    std::vector<uint64_t> expectShape{2, 1};
    EXPECT_EQ(expectShape, res.shape());

    std::vector<float> expectData{3, 3};
    for (std::size_t i = 0; i < res.size(); ++i) {
        EXPECT_EQ(res.data()[i], expectData[i]);
    }
}