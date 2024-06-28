#include "gtest/gtest.h"
#include "../tensor.h"


TEST(TensorTest, DefaultConstructor) {
    std::vector<float> data{};
    std::vector<uint64_t> shape{};

    Tensor<float> t{};
    EXPECT_EQ(t.data(), data);
    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.size(), 0);
}

TEST(TensorTest, CopyInitializationConstructor) {
    std::vector<float> data{};
    std::vector<uint64_t> shape{};

    Tensor<float> t = Tensor<float>{};
    EXPECT_EQ(t.data(), data);
    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.size(), 0);
}

TEST(TensorTest, ParametrizedConstructor) {
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint64_t> shape{2, 2};

    Tensor<float> t{{1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}};
    EXPECT_EQ(t.data(), data);
    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.size(), 4);
}

TEST(TensorTest, ParametrizedCopyConstructor) {
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint64_t> shape{2, 2};

    Tensor<float> t = Tensor<float>{{1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}};
    EXPECT_EQ(t.data(), data);
    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.size(), 4);
}


TEST(TensorTest, CopyConstructor) {
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint64_t> shape{2, 2};

    Tensor<float> t1 = Tensor<float>{{1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}};
    Tensor<float> t2(t1);
    EXPECT_EQ(t2.data(), data);
    EXPECT_EQ(t2.shape(), shape);
    EXPECT_EQ(t2.size(), 4);
}

TEST(TensorTest, MoveConstructor) {
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint64_t> shape{2, 2};

    Tensor<float> t1{data, shape}; // Original tensor

    // Move t1 into t2
    Tensor<float> t2(std::move(t1)); 

    // Assertions
    EXPECT_EQ(t2.data(), data);     // t2 should have the original data
    EXPECT_EQ(t2.shape(), shape);   // t2 should have the original shape
    EXPECT_EQ(t2.size(), 4);        // t2 should have the correct size

    // Check that t1 is in a valid but moved-from state
    EXPECT_TRUE(t1.data().empty()); // t1's data should be empty (moved)
    // You might want additional checks for t1's shape or other internal state
}