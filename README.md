# C++ Inference Engine from scratch

I am developing this project to learn C++ and get hands-on experience with inference engines.

## How to build

1. clone the project: `git clone git@github.com:MichalPitr/inference_engine.git`
2. `cd inference_engine`
3. `sh build.sh`

CMake will complain if you are missing some system dependencies: protobuf, gtest, google benchmark, yaml-cpp

## How to run simple example

This starts up an http server and uses python to send requests. You can also do the equivalent with curl via command line.

1. Build like explained above.
2. `cd server`
3. `go run main.go`
4. Open another terminal
5. `cd utils`
6. `source venv/bin/activate`
7. `python infer_server.py`

## Backlog:

* Optimize Cuda kernels. Gemm is very naive at the moment.
* Add dynamic batching to Go server.
* Add graph optimizations.
* Add input validations to Go server.
* Optimize memory allocator usage - should check available memory during loading, total memory usage can be pretty accurately estimated.
* Improve error handling.
* Explore NVTX profiling.

## Contributing

This project wasn't designed with the idea of external contributions but if you fancy, improvements are welcome!

## Blog posts

I enjoy writing technical blog posts and I've written some about this project:

* [Initial design](https://michalpitr.substack.com/p/build-your-own-inference-engine-from)
* [Profiling-driven optimizations](https://michalpitr.substack.com/p/inference-engine-optimizing-performance)
