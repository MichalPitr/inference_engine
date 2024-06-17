#include "gtest/gtest.h"
#include "../node.h" 

TEST(NodeTest, Constructor) {
    // Tensor setup
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint64_t> shape = {2, 2};
    Node node("relu1", OpType::Relu);
    std::vector<std::string> inputs {"A", "B", "bias"};
    node.addInput(inputs[0]);
    node.addInput(inputs[1]);
    node.addInput(inputs[2]);

    EXPECT_EQ(node.getName(), "relu1");
    EXPECT_EQ(node.getInputs(), inputs);
}
