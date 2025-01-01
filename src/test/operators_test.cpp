#include "../operators.h"

#include "../tensor.h"
#include "gtest/gtest.h"

TEST(OperatorsTest, flatten) {
    const float data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<size_t> shape = {1, 1, 2, 4};
    Tensor<float> t1(data, shape);

    // Note: flatten modifies the original tensor in-place.
    Tensor<float> t2 = CpuOperators<float>::flatten(t1, uint64_t{3});
    std::vector<uint64_t> expectedShape1{2, 4};
    EXPECT_EQ(expectedShape1, t2.shape());

    Tensor<float> t3 = CpuOperators<float>::flatten(t2, uint64_t{0});
    std::vector<uint64_t> expectedShape2{1, 8};
    EXPECT_EQ(expectedShape2, t3.shape());
}

TEST(OperatorsTest, relu) {
    const float data1[] = {-1.0f, 0, 1, -1, 1, -0.5f, -0, 0.5f};
    std::vector<size_t> shape1 = {2, 4};
    Tensor<float> t1(data1, shape1);
    auto t2 = CpuOperators<float>::relu(t1);
    std::vector<float> expected{0, 0, 1, 0, 1, 0, 0, 0.5};

    for (std::size_t i = 0; i < t2.size(); ++i) {
        EXPECT_EQ(t2.data()[i], expected[i]);
    }

    const float data2[] = {1, 1, 1, 1};
    std::vector<size_t> shape2 = {2, 2};
    Tensor<float> t3(data2, shape2);

    auto t4 = CpuOperators<float>::relu(t3);
    std::vector<float> expected2{1, 1, 1, 1};
    for (std::size_t i = 0; i < t4.size(); ++i) {
        EXPECT_EQ(t4.data()[i], expected2[i]);
    }
}

TEST(OperatorsTest, gemmMatrixVector) {
    const float dataA[] = {1, 2, 3, 4};
    std::vector<size_t> shapeA = {2, 2};
    Tensor<float> A(dataA, shapeA);

    const float dataB[] = {1, 1};
    std::vector<size_t> shapeB = {2, 1};
    Tensor<float> B(dataB, shapeB);

    const float dataBias[] = {1, 1};
    std::vector<size_t> shapeBias = {2, 1};
    Tensor<float> bias(dataBias, shapeBias);

    auto res = CpuOperators<float>::gemm(A, B, bias, false, false, 1, 1);

    std::vector<uint64_t> expectShape{2, 1};
    EXPECT_EQ(expectShape, res.shape());
    std::vector<float> expectData{4, 8};
    for (std::size_t i = 0; i < res.size(); ++i) {
        EXPECT_EQ(res.data()[i], expectData[i]);
    }
}

TEST(OperatorsTest, gemmMatrixMatrix) {
    const float dataA[] = {1, 2, 3, 4};
    std::vector<size_t> shapeA = {2, 2};
    Tensor<float> A(dataA, shapeA);

    const float dataB[] = {1, 1, 1, 1};
    std::vector<size_t> shapeB = {2, 2};
    Tensor<float> B(dataB, shapeB);

    const float dataBias[] = {1, 2};
    std::vector<size_t> shapeBias = {2, 1};
    Tensor<float> bias(dataBias, shapeBias);

    auto res = CpuOperators<float>::gemm(A, B, bias, false, false, 1, 1);

    std::vector<uint64_t> expectShape{2, 2};
    EXPECT_EQ(expectShape, res.shape());

    std::vector<float> expectData{4, 4, 9, 9};
    for (std::size_t i = 0; i < res.size(); ++i) {
        EXPECT_EQ(res.data()[i], expectData[i]);
    }
}

TEST(OperatorsTest, gemmMatrixMatrixTransA) {
    const float dataA[] = {1, 2};
    std::vector<size_t> shapeA = {2, 1};
    Tensor<float> A(dataA, shapeA);

    const float dataB[] = {1, 1, 1, 1};
    std::vector<size_t> shapeB = {2, 2};
    Tensor<float> B(dataB, shapeB);

    const float dataBias[] = {0, 0};
    std::vector<size_t> shapeBias = {2};
    Tensor<float> bias(dataBias, shapeBias);

    auto res = CpuOperators<float>::gemm(A, B, bias, true, false, 1, 1);

    std::vector<uint64_t> expectShape{1, 2};
    EXPECT_EQ(expectShape, res.shape());

    std::vector<float> expectData{3, 3};
    for (std::size_t i = 0; i < res.size(); ++i) {
        EXPECT_EQ(res.data()[i], expectData[i]);
    }
}

TEST(OperatorsTest, gemmMatrixMatrixTransB) {
    const float dataA[] = {1, 1, 1, 1};
    std::vector<size_t> shapeA = {2, 2};
    Tensor<float> A(dataA, shapeA);

    const float dataB[] = {1, 2};
    std::vector<size_t> shapeB = {1, 2};
    Tensor<float> B(dataB, shapeB);

    const float dataBias[] = {0, 0};
    std::vector<size_t> shapeBias = {2};
    Tensor<float> bias(dataBias, shapeBias);

    auto res = CpuOperators<float>::gemm(A, B, bias, false, true, 1, 1);

    std::vector<uint64_t> expectShape{2, 1};
    EXPECT_EQ(expectShape, res.shape());

    std::vector<float> expectData{3, 3};
    for (std::size_t i = 0; i < res.size(); ++i) {
        EXPECT_EQ(res.data()[i], expectData[i]);
    }
}