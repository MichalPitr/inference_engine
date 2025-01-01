ARG CUDA_VERSION=12.6.1
ARG CUDNN_VERSION=9.5.0.50
ARG OS=ubuntu24.04

# Start with NVIDIA CUDA base image
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-${OS}

# Avoid interactive dialog during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential development tools and libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    vim \
    gdb \
    openssh-client \
    curl \
    protobuf-compiler \
    libbenchmark-dev \
    libgtest-dev \
    libyaml-cpp-dev \
    clang \
    clang-format \
    clang-tidy \
    libc++-dev \
    libc++abi-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CUDA
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Set clang as the default C/C++ compiler
ENV CC=clang
ENV CXX=clang++

WORKDIR /code

CMD ["/bin/bash"]