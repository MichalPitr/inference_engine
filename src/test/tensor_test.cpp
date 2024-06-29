#include "gtest/gtest.h"
#include "../tensor.h"

// Constructors

TEST(TensorTest, DefaultConstructor) {
    std::vector<float> data{};
    std::vector<uint64_t> shape{};

    Tensor<float> t{};
    EXPECT_EQ(t.data(), data);
    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.size(), 0);
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

TEST(TensorTest, CopyAssignmentConstructor) {
    std::vector<float> data{};
    std::vector<uint64_t> shape{};

    Tensor<float> t = Tensor<float>{};
    EXPECT_EQ(t.data(), data);
    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.size(), 0);
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

TEST(TensorTest, MoveAssignmentOperator) {
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint64_t> shape{2, 2};

    Tensor<float> t1{data, shape}; // Original tensor
    Tensor<float> t2;             // Empty tensor

    // Move assign t1 into t2
    t2 = std::move(t1); 

    // Assertions
    EXPECT_EQ(t2.data(), data);     // t2 should have the original data
    EXPECT_EQ(t2.shape(), shape);   // t2 should have the original shape
    EXPECT_EQ(t2.size(), 4);        // t2 should have the correct size

    // Check that t1 is in a valid but moved-from state
    EXPECT_TRUE(t1.data().empty()); // t1's data should be empty (moved)
    // You might want additional checks for t1's shape or other internal state
}

TEST(TensorTest, ParametrizedConstructor) {
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint64_t> shape{2, 2};

    Tensor<float> t{{1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}};
    EXPECT_EQ(t.data(), data);
    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.size(), 4);
}

TEST(TensorTest, ParametrizedCopyAssignmentConstructor) {
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint64_t> shape{2, 2};

    Tensor<float> t = Tensor<float>{{1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}};
    EXPECT_EQ(t.data(), data);
    EXPECT_EQ(t.shape(), shape);
    EXPECT_EQ(t.size(), 4);
}

// Member methods

TEST(TensorTest, RawData) {
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint64_t> shape{2, 2};

    Tensor<float> t{{1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}};
    float *raw_data = t.raw_data();
    EXPECT_NE(raw_data, nullptr);

    // Check if the raw data values are correct
    for (std::size_t i = 0; i < t.size(); ++i) {
        EXPECT_EQ(raw_data[i], data[i]); 
    }

    // Test if modifying the raw data affects the tensor
    raw_data[0] = 5.0f;
    EXPECT_EQ(t.data()[0], 5.0f); 
}
