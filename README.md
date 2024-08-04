# C++ Inference Engine from scratch

This is a project I built to learn more about inference engines and C++. You can learn more about this project in a [blog post](https://michalpitr.substack.com/p/build-your-own-inference-engine-from) I wrote about it!

## How to build

1. clone the project: `git clone git@github.com:MichalPitr/inference_engine.git`
2. `cd inference_engine`
3. `mkdir build`
4. `cd build`
5. `cmake ..`
6. `make .`

CMake will complain if you are missing some system dependencies: protobuf, gtest, google benchmark.

## Run inference

`/<path>/inference_engine/build/src/engine_exe /<path>/inference_engine/models/mnist_ffn_complex.onnx /<path>/inference_engine/inputs/<image>`
