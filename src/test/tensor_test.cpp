#include "../tensor.h"

#include "gtest/gtest.h"

TEST(TensorTest, DefaultConstructor) {
    float *data{};
    std::vector<uint64_t> shape{};

    Tensor<float> t{};
    EXPECT_EQ(t.data(), data);
    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.size(), 0);
}

TEST(TensorTest, CopyConstructor) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint64_t> shape{2, 2};

    Tensor<float> t1 = Tensor<float>{{1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}};
    Tensor<float> t2(t1);
    for (std::size_t i = 0; i < t2.size(); ++i) {
        EXPECT_EQ(t2.data()[i], data[i]);
    }

    EXPECT_EQ(t2.shape(), shape);
    EXPECT_EQ(t2.size(), 4);
}

TEST(TensorTest, CopyAssignmentConstructor) {
    float *data{};
    std::vector<uint64_t> shape{};

    Tensor<float> t = Tensor<float>{};
    for (std::size_t i = 0; i < t.size(); ++i) {
        EXPECT_EQ(t.data()[i], data[i]);
    }
    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.size(), 0);
}

TEST(TensorTest, MoveConstructor) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint64_t> shape{2, 2};

    Tensor<float> t1{data, shape, DeviceType::CPU};

    Tensor<float> t2(std::move(t1));

    for (std::size_t i = 0; i < t2.size(); ++i) {
        EXPECT_EQ(t2.data()[i], data[i]);
    }
    EXPECT_EQ(t2.shape(), shape);
    EXPECT_EQ(t2.size(), 4);

    EXPECT_TRUE(t1.data() == nullptr);
}

TEST(TensorTest, MoveAssignmentOperator) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint64_t> shape{2, 2};

    Tensor<float> t1{data, shape};
    Tensor<float> t2;

    t2 = std::move(t1);

    for (std::size_t i = 0; i < t2.size(); ++i) {
        EXPECT_EQ(t2.data()[i], data[i]);
    }
    EXPECT_EQ(t2.shape(), shape);
    EXPECT_EQ(t2.size(), 4);

    EXPECT_TRUE(t1.data() == nullptr);
}

TEST(TensorTest, ParametrizedConstructor) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};

    std::vector<uint64_t> shape{2, 2};

    Tensor<float> t{{1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}};
    for (std::size_t i = 0; i < t.size(); ++i) {
        EXPECT_EQ(t.data()[i], data[i]);
    }
    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.size(), 4);
}

TEST(TensorTest, ParametrizedCopyAssignmentConstructor) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint64_t> shape{2, 2};

    Tensor<float> t = Tensor<float>{{1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}};
    for (std::size_t i = 0; i < t.size(); ++i) {
        EXPECT_EQ(t.data()[i], data[i]);
    }
    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.size(), 4);
}
