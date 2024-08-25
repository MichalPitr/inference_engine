#include <iostream>
#include <sstream>

#include "input_loader.h"
#include "model_loader.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <input.data>"
                  << std::endl;
        return 1;
    }

    const std::string modelFile = argv[1];
    const std::string inputFile = argv[2];

    try {
        ModelLoader loader;
        auto engine = loader.load(modelFile);
        for (int i = 0; i <= 1000; ++i) {
            auto input = load_input(inputFile);
            auto output = engine->infer(input);
            std::cout << "Out: " << output.to_string() << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}