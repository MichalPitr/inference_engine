#include "gtest/gtest.h"
#include "../node.h" 

TEST(NodeTest, Constructor) {
    // Tensor setup
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint64_t> shape = {2, 2};
    Tensor* tensor = new Tensor(data, shape, DataType::FLOAT32);

    Node node("relu1", OpType::Relu, tensor);

    EXPECT_EQ(node.getName(), "relu1");
}
