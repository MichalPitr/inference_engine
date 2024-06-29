
#include <iostream>
#include <assert.h>

#include "gemm.h"
#include "operators.h"

// Tensor* gemm(std::vector<const Tensor&> &inputs) {
//     std::cout << "Op: Gemm" << std::endl;

//     if (inputs.size() != 3) {
//         throw std::runtime_error("Gemm operator expects exactly three input tensors.");
//     }

//     const auto* A = inputs[0];
//     const auto* B = inputs[1];
//     const auto* C = inputs[2];

//     // Input Validation
//     if (A.shape().size() != 2 || B.shape().size() != 2 || C.shape().size() != 1) {
//         std::cerr << "A dims: " << A.shape().size() << "B dims" << B.shape().size() << "C dims" << C.shape().size() << std::endl;
//         throw std::invalid_argument("Invalid dimensions for Gemm inputs.");
//     }
//     if (A.shape()[1] != B.shape()[1]) {
//         throw std::invalid_argument("Matrix dimensions are not compatible for multiplication in Gemm.");
//     }
//     if (B.shape()[0] != C.shape()[0]) {
//         throw std::invalid_argument("Bias dimensions are not compatible with the result in Gemm.");
//     }

//     std::cout << "A.shape = (" << A.shape()[0] << ", " << A.shape()[1] << ")" << std::endl;
//     std::cout << "B.shape = (" << B.shape()[0] << ", " << B.shape()[1] << ")" << std::endl;
//     std::cout << "C.shape = (" << C.shape()[0] << ")" << std::endl;


//     // Calculate output dimensions
//     uint64_t N = A.shape()[1];
//     uint64_t M = B.shape()[0];
//     uint64_t K = A.shape()[0];
//     std::cout << "N: " << N << std::endl;
//     std::cout << "M: " << M << std::endl;
//     std::cout << "K: " << K << std::endl;

//     std::vector<uint64_t> dims = {K, M};

//     // Allocate memory for output and copy bias (C) using a loop
//     std::vector<float> outData(M);
//     std::cout << "outData size " << outData.size() << std::endl;

//     // Perform GEMM operation
//     // Pass raw pointers to the underlying `gemm` function
//     const float* AData = reinterpret_cast<const float*>(A.data().data());
//     const float* BData = reinterpret_cast<const float*>(B.data().data());
//     const float* CData = reinterpret_cast<const float*>(C.data().data());

//     std::cout << "Running gemm" << std::endl;
//     gemm(BData, AData, CData, outData.data(), M, N, K); // Assuming your gemm function is modified to accept raw pointers
//     std::cout << "finished gemm" << std::endl;

//     Tensor* result = new Tensor(outData, dims, DataType::FLOAT32);

//     // Print out values
//     std::cout << "out: ";
//     for (std::size_t i = 0; i < outData.size(); ++i) {
//         std::cout << outData[i] << ", ";
//     }
//     std::cout << std::endl;

//     return result;
// }

// flatten returns a new flattened version of node. Caller is responsible for managing memory.
template <typename T>
Tensor<T> flatten(Tensor<T>& tensor, uint64_t axis)
{
    std::cout << "Op: Flatten" << std::endl;

    // assert(tensor.shape().size() == 4);
    assert(axis <= tensor.shape().size());
    
    // Mnist input only
    // assert(tensor.shape()[0] == 1); // batch size
    // assert(tensor.shape()[1] == 1); // channels
    // assert(tensor.shape()[2] == 28); 
    // assert(tensor.shape()[3] == 28);

    uint64_t dimBefore = 1;
    for (std::size_t i = 0; i < axis; ++i) {
        dimBefore *= tensor.shape()[i];
    }
    std::cout << "dim before: " << dimBefore << std::endl;

    uint64_t dimAfter = 1;
    for (std::size_t i = axis; i < tensor.shape().size(); ++i) {
        dimAfter *= tensor.shape()[i];
    }
    std::cout << "dim after: " << dimAfter << std::endl;
    
    // copy initialize. Would be better if we could modify it in place, but we 
    // don't know if some other function relies on the input tensor. If we can do some dependency analysis, we could
    // probably optimize this.
    Tensor<T> flat(tensor);
    for (auto s: flat.shape()) {
        std::cout << "shape dim: " << s << '\n';
    }
    flat.setShape({dimBefore, dimAfter});
    for (auto s: flat.shape()) {
        std::cout << "shape after dim: " << s << '\n';
    }
    // Diagnostic printing
    std::cout << "flatten out:";
    for (std::size_t i = 0; i < flat.size(); ++i)
    {
        std::cout << " " << flat.data()[i];
    }
    std::cout << std::endl;

    // Set tensor name.
    return flat;
}

// // relu
// Tensor* relu(std::vector<const Tensor*> &inputs)
// {
//     std::cout << "Op: Relu" << std::endl;
//     assert(inputs.size() == 1);
//     const auto &tensor = *inputs[0];

//     if (tensor.dataType() == DataType::FLOAT32 && !tensor.data().empty())
//     {
//         // Copy input data.
//         std::vector<float> outputData(tensor.data());
//         for (std::size_t i = 0; i < outputData.size(); ++i)
//         {
//             outputData[i] = std::max(0.0f, outputData[i]);
//         }

//         // Print the modified output values (optional)
//         std::cout << "ReLU: ";
//         assert(tensor.shape().size() == 2);
//         for (uint64_t i = 0; i < tensor.shape()[1]; ++i)
//         { // Assuming the second dimension is the relevant one
//             std::cout << outputData[i] << ", ";
//         }
//         std::cout << std::endl;
//         return new Tensor(outputData, tensor.shape(), DataType::FLOAT32);
//     }
//     else
//     {
//         std::cerr << "Unsupported data type or empty raw data in ReLU" << std::endl;
//         exit(1);
//     }
// }

template Tensor<float> flatten<float>(Tensor<float>& tensor, uint64_t axis); 
