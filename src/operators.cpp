
#include <iostream>
#include <assert.h>

#include "gemm.h"
#include "operators.h"

Tensor* gemm(std::vector<const Tensor*> &inputs) {
    std::cout << "Op: Gemm" << std::endl;

    if (inputs.size() != 3) {
        throw std::runtime_error("Gemm operator expects exactly three input tensors.");
    }

    const auto* A = inputs[0];
    const auto* B = inputs[1];
    const auto* C = inputs[2];

    // Input Validation
    if (A->getShape().size() != 2 || B->getShape().size() != 2 || C->getShape().size() != 1) {
        std::cerr << "A dims: " << A->getShape().size() << "B dims" << B->getShape().size() << "C dims" << C->getShape().size() << std::endl;
        throw std::invalid_argument("Invalid dimensions for Gemm inputs.");
    }
    if (A->getShape()[1] != B->getShape()[1]) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication in Gemm.");
    }
    if (B->getShape()[0] != C->getShape()[0]) {
        throw std::invalid_argument("Bias dimensions are not compatible with the result in Gemm.");
    }

    std::cout << "A.shape = (" << A->getShape()[0] << ", " << A->getShape()[1] << ")" << std::endl;
    std::cout << "B.shape = (" << B->getShape()[0] << ", " << B->getShape()[1] << ")" << std::endl;
    std::cout << "C.shape = (" << C->getShape()[0] << ")" << std::endl;


    // Calculate output dimensions
    uint64_t N = A->getShape()[1];
    uint64_t M = B->getShape()[0];
    uint64_t K = A->getShape()[0];
    std::cout << "N: " << N << std::endl;
    std::cout << "M: " << M << std::endl;
    std::cout << "K: " << K << std::endl;


    std::vector<uint64_t> dims = {K, M};

    // Allocate memory for output and copy bias (C) using a loop
    std::vector<float> outData(M);
    std::cout << "outData size " << outData.size() << std::endl;

    // Perform GEMM operation
    // Pass raw pointers to the underlying `gemm` function
    const float* AData = reinterpret_cast<const float*>(A->getData().data());
    const float* BData = reinterpret_cast<const float*>(B->getData().data());
    const float* CData = reinterpret_cast<const float*>(C->getData().data());

    std::cout << "Running gemm" << std::endl;
    gemm(BData, AData, CData, outData.data(), M, N, K); // Assuming your gemm function is modified to accept raw pointers
    std::cout << "finished gemm" << std::endl;

    Tensor* result = new Tensor(outData, dims, DataType::FLOAT32);

    // Print out values
    std::cout << "out: ";
    for (std::size_t i = 0; i < outData.size(); ++i) {
        std::cout << outData[i] << ", ";
    }
    std::cout << std::endl;

    return result;
}

// flatten returns a new flattened version of node. Caller is responsible for managing memory.
Tensor *flatten(std::vector<const Tensor*> &inputs, uint64_t axis)
{
    std::cout << "Op: Flatten" << std::endl;
    if (inputs.size() != 1)
    {
        throw std::runtime_error("Expected inputs.size to be 1 in flatten operation.");
    }

    const auto inputTensor = inputs[0];

    assert(inputTensor->getShape().size() == 4);
    assert(axis <= inputTensor->getShape().size());
    
    // Mnist input only
    assert(inputTensor->getShape()[0] == 1); // batch size
    assert(inputTensor->getShape()[1] == 1); // channels
    assert(inputTensor->getShape()[2] == 28); 
    assert(inputTensor->getShape()[3] == 28);

    uint64_t dimBefore = 1;
    for (std::size_t i = 0; i < axis; ++i) {
        dimBefore *= inputTensor->getShape()[i];
    }
    std::cout << "dim before: " << dimBefore << std::endl;

    uint64_t dimAfter = 1;
    for (std::size_t i = axis; i < inputTensor->getShape().size(); ++i) {
        dimAfter *= inputTensor->getShape()[i];
    }
    std::cout << "dim after: " << dimAfter << std::endl;
    
    std::vector<float> flattened(inputTensor->getData()); 
    Tensor* flat = new Tensor(flattened, std::vector<uint64_t> {dimBefore, dimAfter}, DataType::FLOAT32);
    
    // Diagnostic printing
    std::cout << "flatten.size(): " << flattened.size() << std::endl;
    std::cout << "flatten out:";
    for (std::size_t i = 0; i < flattened.size(); ++i)
    {
        std::cout << " " << flattened[i];
    }
    std::cout << std::endl;

    // Set tensor name.
    return flat;
}

// relu
Tensor* relu(std::vector<const Tensor*> &inputs)
{
    std::cout << "Op: Relu" << std::endl;
    assert(inputs.size() == 1);
    const auto &inputTensor = *inputs[0];

    if (inputTensor.getDataType() == DataType::FLOAT32 && !inputTensor.getData().empty())
    {
        // Copy input data.
        std::vector<float> outputData(inputTensor.getData());
        for (std::size_t i = 0; i < outputData.size(); ++i)
        {
            outputData[i] = std::max(0.0f, outputData[i]);
        }

        // Print the modified output values (optional)
        std::cout << "ReLU: ";
        assert(inputTensor.getShape().size() == 2);
        for (uint64_t i = 0; i < inputTensor.getShape()[1]; ++i)
        { // Assuming the second dimension is the relevant one
            std::cout << outputData[i] << ", ";
        }
        std::cout << std::endl;
        return new Tensor(outputData, inputTensor.getShape(), DataType::FLOAT32);
    }
    else
    {
        std::cerr << "Unsupported data type or empty raw data in ReLU" << std::endl;
        exit(1);
    }
}