#include "../tensor.h"

#include "gtest/gtest.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>

#include "../cuda_allocator.h"
#endif

TEST(TensorTest, ConstructorWithShape) {
    Tensor<float> tensor({2, 3, 4});
    EXPECT_EQ(tensor.shape(), (std::vector<size_t>{2, 3, 4}));
    EXPECT_EQ(tensor.size(), 24);
    EXPECT_EQ(tensor.device(), DeviceType::CPU);
    EXPECT_NE(tensor.data(), nullptr);
}

TEST(TensorTest, ConstructorWithData) {
    std::vector<float> data(24, 1.0f);
    Tensor<float> tensor(data.data(), {2, 3, 4});
    EXPECT_EQ(tensor.shape(), (std::vector<size_t>{2, 3, 4}));
    EXPECT_EQ(tensor.size(), 24);
    EXPECT_EQ(tensor.device(), DeviceType::CPU);
    EXPECT_NE(tensor.data(), nullptr);

    // Check if data was correctly copied
    for (size_t i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(tensor.data()[i], 1.0f);
    }
}

TEST(TensorTest, CopyConstructor) {
    Tensor<float> original({2, 3, 4});
    for (size_t i = 0; i < original.size(); ++i) {
        original.data()[i] = static_cast<float>(i);
    }

    Tensor<float> copy(original);
    EXPECT_EQ(copy.shape(), original.shape());
    EXPECT_EQ(copy.size(), original.size());
    EXPECT_EQ(copy.device(), original.device());
    EXPECT_NE(copy.data(), original.data());  // Ensure deep copy

    for (size_t i = 0; i < copy.size(); ++i) {
        EXPECT_FLOAT_EQ(copy.data()[i], original.data()[i]);
    }
}

TEST(TensorTest, MoveConstructor) {
    Tensor<float> original({2, 3, 4});
    float* originalData = original.data();

    Tensor<float> moved(std::move(original));
    EXPECT_EQ(moved.shape(), (std::vector<size_t>{2, 3, 4}));
    EXPECT_EQ(moved.size(), 24);
    EXPECT_EQ(moved.device(), DeviceType::CPU);
    EXPECT_EQ(moved.data(), originalData);

    // Check that the original tensor has been properly moved from
    EXPECT_EQ(original.data(), nullptr);
    EXPECT_TRUE(original.shape().empty());
    EXPECT_EQ(original.size(), 0);
}

TEST(TensorTest, CopyAssignment) {
    Tensor<float> original({2, 3, 4});
    for (size_t i = 0; i < original.size(); ++i) {
        original.data()[i] = static_cast<float>(i);
    }

    Tensor<float> copy({1, 1, 1});  // Different size
    copy = original;

    EXPECT_EQ(copy.shape(), original.shape());
    EXPECT_EQ(copy.size(), original.size());
    EXPECT_EQ(copy.device(), original.device());
    EXPECT_NE(copy.data(), original.data());  // Ensure deep copy

    for (size_t i = 0; i < copy.size(); ++i) {
        EXPECT_FLOAT_EQ(copy.data()[i], original.data()[i]);
    }
}

TEST(TensorTest, MoveAssignment) {
    Tensor<float> original({2, 3, 4});
    float* originalData = original.data();

    Tensor<float> moved({1, 1, 1});  // Different size
    moved = std::move(original);

    EXPECT_EQ(moved.shape(), (std::vector<size_t>{2, 3, 4}));
    EXPECT_EQ(moved.size(), 24);
    EXPECT_EQ(moved.device(), DeviceType::CPU);
    EXPECT_EQ(moved.data(), originalData);

    // Check that the original tensor has been properly moved from
    EXPECT_EQ(original.data(), nullptr);
    EXPECT_TRUE(original.shape().empty());
    EXPECT_EQ(original.size(), 0);
}

#ifdef USE_CUDA
TEST(TensorTest, DeviceTransferCPUtoCUDA) {
    Tensor<float> cpuTensor({2, 3, 4});
    for (size_t i = 0; i < cpuTensor.size(); ++i) {
        cpuTensor.data()[i] = static_cast<float>(i);
    }
    auto cudaAllocator = std::make_shared<CudaAllocator>();
    cpuTensor.to(DeviceType::CUDA, cudaAllocator);
    EXPECT_EQ(cpuTensor.device(), DeviceType::CUDA);

    // Create a new CPU tensor to check data
    Tensor<float> checkTensor({2, 3, 4});
    cudaMemcpy(checkTensor.data(), cpuTensor.data(),
               cpuTensor.size() * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < checkTensor.size(); ++i) {
        EXPECT_FLOAT_EQ(checkTensor.data()[i], static_cast<float>(i));
    }
}

TEST(TensorTest, DeviceTransferCUDAtoCPU) {
    Tensor<float> cudaTensor({2, 3, 4}, std::make_shared<CudaAllocator>());
    std::vector<float> init_data(cudaTensor.size());
    for (size_t i = 0; i < init_data.size(); ++i) {
        init_data[i] = static_cast<float>(i);
    }
    cudaMemcpy(cudaTensor.data(), init_data.data(),
               cudaTensor.size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaTensor.to(DeviceType::CPU);
    EXPECT_EQ(cudaTensor.device(), DeviceType::CPU);

    for (size_t i = 0; i < cudaTensor.size(); ++i) {
        EXPECT_FLOAT_EQ(cudaTensor.data()[i], static_cast<float>(i));
    }
}
#endif

TEST(TensorTest, InvalidDeviceTransfer) {
    Tensor<float> tensor({2, 3, 4});
    EXPECT_THROW(tensor.to(static_cast<DeviceType>(99)), std::runtime_error);
}
