# C++ Inference Engine from scratch

I am developing this project to learn C++ and get hands-on experience with inference engines.

## How to build

1. clone the project: `git clone git@github.com:MichalPitr/inference_engine.git`
2. `cd inference_engine`
3. `mkdir build`
4. `cd build`
5. `cmake ..`
6. `make .`

CMake will complain if you are missing some system dependencies: protobuf, gtest, google benchmark, yaml-cpp

## Run inference with a sample image

`/${project_root}/build/src/engine_exe ${project_root}/model_repository/mnist.yaml ${project_root}/inputs/image_0.ubyte`

## Backlog:

* Optimize Cuda kernels.
* Add dynamic batching in Go server.
* Add graph optimizations.

## Contributing

This project wasn't designed with the idea of external contributions but if you fancy, improvements are welcome!

## Blog posts

I enjoy writing technical blog posts and I've written some about this project:

* [Initial design](https://michalpitr.substack.com/p/build-your-own-inference-engine-from)
* [Profiling-driven optimizations](https://michalpitr.substack.com/p/inference-engine-optimizing-performance)
