#include "gtest/gtest.h"
#include "../tensor.h"

// Test Fixture for Tensor
class TensorTest : public ::testing::Test {
protected:
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint64_t> shape = {2, 2};
    DataType dataType = DataType::FLOAT32;
    Tensor tensor = Tensor(data, shape, dataType);
};

// Constructor Test
TEST_F(TensorTest, ConstructorValidInput) {
    EXPECT_EQ(tensor.getData(), data);
    EXPECT_EQ(tensor.getShape(), shape);
    EXPECT_EQ(tensor.getDataType(), dataType);
    EXPECT_EQ(tensor.getNumElements(), 4);
}

// Constructor Test: Invalid Data Type
TEST_F(TensorTest, ConstructorInvalidDataType) {
    EXPECT_THROW(Tensor(data, shape, DataType::UNKNOWN), std::runtime_error); 
}

// Constructor Test: Mismatched Shape and Data
TEST_F(TensorTest, ConstructorMismatchedShapeAndData) {
    std::vector<uint64_t> wrongShape = {2, 3}; 
    EXPECT_THROW(Tensor(data, wrongShape, dataType), std::runtime_error); 
}

// Data Access Test
TEST_F(TensorTest, DataAccessors) {
    EXPECT_EQ(tensor.getData()[0], 1.0f);  
    EXPECT_EQ(tensor.getData()[3], 4.0f); 
    EXPECT_EQ(tensor.getShape()[0], 2);    
    EXPECT_EQ(tensor.getShape()[1], 2);    
}