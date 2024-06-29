#include "gtest/gtest.h"
#include "../node.h" 

TEST(NodeTest, Constructor) {
    // Tensor setup
    Node node("relu1", OpType::Relu);
    std::vector<std::string> inputs {"A", "B", "bias"};
    node.addInput(inputs[0]);
    node.addInput(inputs[1]);
    node.addInput(inputs[2]);

    EXPECT_EQ(node.getName(), "relu1");
    EXPECT_EQ(node.getInputs(), inputs);
}
