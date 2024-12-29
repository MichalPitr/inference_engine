#include <cuda_runtime.h>

#include <cassert>
#include <iostream>
#include <stdexcept>

const std::size_t BLOCK_SIZE{16};

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

/*
Tensors are of shape:
A: (n, m)
B: (m, k)
C: (n, k)
*/
__global__ void gemm_kernel_naive(const float *A, const float *B,
                                  const float *bias, float *out, int n, int m,
                                  int k, bool transA, bool transB, float alpha,
                                  float beta) {
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < k) {
        float res = 0.0f;

        for (int i = 0; i < m; ++i) {
            float aVal = transA ? A[i * n + row] : A[row * m + i];
            float bVal = transB ? B[col * m + i] : B[i * k + col];
            res += aVal * bVal;
        }
        out[row * k + col] = res * alpha + bias[col] * beta;
    }
}

template <const uint BLOCKSIZE>
__global__ void gemm_kernel_tiled(const float *A, const float *B,
                                  const float *bias, float *out, int n, int m,
                                  int k, bool transA, bool transB, float alpha,
                                  float beta) {
    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    uint tx = threadIdx.x;
    uint ty = threadIdx.y;

    int row = by * BLOCKSIZE + ty;
    int col = bx * BLOCKSIZE + tx;

    // Calcualtes single entry of C per block.
    float res = 0.0f;

    for (int blkIdx = 0; blkIdx < CEIL_DIV(m, BLOCKSIZE); ++blkIdx) {
        // Collaboratively load tile
        if (row < n && blkIdx * BLOCKSIZE + tx < m) {
            // handle transpose.
            As[ty][tx] = transA ? A[(blkIdx * BLOCKSIZE + tx) * n + row]
                                : A[row * m + (blkIdx * BLOCKSIZE + tx)];
        } else {
            // Out of bounds defaults to 0.
            As[ty][tx] = 0.0f;
        }

        if (blkIdx * BLOCKSIZE + ty < m && col < k) {
            // Handle transpose.
            Bs[ty][tx] = transB ? B[col * m + (blkIdx * BLOCKSIZE + ty)]
                                : B[(blkIdx * BLOCKSIZE + ty) * k + col];
        } else {
            // Out of bounds defaults to 0.
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Matmul over tile
        for (int i = 0; i < BLOCKSIZE; ++i) {
            res += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < n && col < k) {
        out[row * k + col] = res * alpha + bias[col] * beta;
    }
}

/*
    Tensors are of shape:
    A: (n, m)
    B: (m, k)
    C: (n, k)
*/
template <const uint BLOCKSIZE>
__global__ void gemm_kernel_tiled_1D(const float *A, const float *B,
                                     const float *bias, float *out, int N,
                                     int M, int K, bool transA, bool transB,
                                     float alpha, float beta) {
    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

    // Block x y coordinates
    uint bx = blockIdx.x;
    uint by = blockIdx.y;

    // thread x y coordinates.
    uint tx = threadIdx.x % BLOCKSIZE;
    uint ty = threadIdx.x / BLOCKSIZE;

    // output row column.
    int row = by * BLOCKSIZE + ty;
    int col = bx * BLOCKSIZE + tx;

    // Calcualtes single entry of C per block.
    float res = 0.0f;

    for (int blkIdx = 0; blkIdx < CEIL_DIV(M, BLOCKSIZE); ++blkIdx) {
        // Collaboratively load tile
        if (row < N && blkIdx * BLOCKSIZE + tx < M) {
            As[ty][tx] = transA ? A[(blkIdx * BLOCKSIZE + tx) * N + row]
                                : A[row * M + (blkIdx * BLOCKSIZE + tx)];
        } else {
            // Out of bounds defaults to 0.
            As[ty][tx] = 0.0f;
        }

        if (blkIdx * BLOCKSIZE + ty < M && col < K) {
            Bs[ty][tx] = transB ? B[col * M + (blkIdx * BLOCKSIZE + ty)]
                                : B[(blkIdx * BLOCKSIZE + ty) * K + col];
        } else {
            // Out of bounds defaults to 0.
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Matmul over tile
        for (int i = 0; i < BLOCKSIZE; ++i) {
            res += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < N && col < K) {
        out[row * K + col] = res * alpha + bias[col] * beta;
    }
}

/** Difference from simple tiled approach:

   This implementation only handles non-transposed matrices as indicated by the
   _nn suffix.

    1. Tiles are not (BLOCKSIZE, BLOCKSIZE), but rather (BN, BM) and (BM,
   BK). Important is that each thread has to process (BN*BK) / ((BN*BM +
   BM+BK)/2) = (2*BN*BK)/(BN*BM + BM+BK) results. So then supposing we use
   (64, 8), (8, 64), we get 2*64*64 / (16*64) = 8.

    2. Each thread still only loads one value, but then computes TM
   outputs. Each thread calculates an (TM,1) vector of C as a column. So thread
   (0,0,0) will calculate C[0][0], C[1][0], ... C[TM][0].

    Tensors are of shape:
        A: (n, m)
        B: (m, k)
        C: (n, k)
        bias: (1, k)
*/
template <const int BN, const int BM, const int BK, const int TM>
__global__ void gemm_kernel_tiled_1D_blocktiling_nn(const float *A,
                                                    const float *B,
                                                    const float *bias, float *C,
                                                    int N, int M, int K,
                                                    float alpha, float beta) {
    __shared__ float As[BN][BM];  // 64, 8
    __shared__ float Bs[BM][BK];  // 8, 64

    // Block x y coordinates
    uint bx = blockIdx.x;  // (0 - K div BK)
    uint by = blockIdx.y;  // (0 - N div BN)

    // Thread for
    const uint threadCol = threadIdx.x % BK;  // range: 0-64
    const uint threadRow = threadIdx.x / BK;  // range: 0-8

    assert(BN * BM == blockDim.x);
    assert(BM * BK == blockDim.x);
    const uint innerColA = threadIdx.x % BM;  // range: 0 - 8
    const uint innerRowA = threadIdx.x / BM;  // range: 0 - 64
    const uint innerColB = threadIdx.x % BK;  // range: 0 - 64
    const uint innerRowB = threadIdx.x / BK;  // range: 0 - 8

    // Each thread block processes one block row of A and block column of B.
    // This means we need to correctly account for offsets depending on in which
    // thread block this kernel is.

    // This says how many rows * width of A to skip.
    // by * BN gives the size of the block tile and M is the row width.
    const uint A_baseoffset = by * BN * M;

    // This says how many columns * height of B to sip.
    // bx * BK gives the number of columns to skip. Using this offset, whenever
    // we index into a row, we'll automatically skip the first offset columns.
    const uint B_baseoffset = bx * BK;

    // Each thread calculates TM results.
    float threadResults[TM] = {0.0};

    // Outer loop over block tiles. Block tile in A is of shape (BN, BM) and
    // block tile in B is of shape (BM, BK). Supposing A and B are both
    // (128, 128), and (BN=64, BM=8), (BM=8, BK=64). Each thread block will
    // calculate a 64x64 block of C. This is done by sliding a (64x8)
    // blocktile across A and B in lock-step. Partial results are
    // accummulated and once fully slided, written to C.
    for (uint blockTileOffset = 0; blockTileOffset < M; blockTileOffset += BM) {
        const bool validRowA = (by * BN + innerRowA) < N;
        const bool validColA = (blockTileOffset + innerColA) < M;
        const uint A_idx =
            A_baseoffset + blockTileOffset + innerRowA * M + innerColA;
        if (validRowA && validColA && A_idx < N * M) {
            As[innerRowA][innerColA] = A[A_idx];
        } else {
            As[innerRowA][innerColA] = 0;
        }

        const bool validRowB = (blockTileOffset + innerRowB) < M;
        const bool validColB = (bx * BK + innerColB) < K;
        const uint B_idx =
            B_baseoffset + blockTileOffset * K + innerColB + innerRowB * K;
        if (validRowB && validColB && B_idx < M * K) {
            Bs[innerRowB][innerColB] = B[B_idx];
        } else {
            Bs[innerRowB][innerColB] = 0;
        }

        __syncthreads();
        // Calculate per-thread results. Each thread calculates TM dot products.
        // This is done by taking TM rows of As and multiplying them with a
        // single column of Bs.
        for (uint resIdx = 0; resIdx < TM; ++resIdx) {
            for (uint dotIdx = 0; dotIdx < BM; ++dotIdx) {
                threadResults[resIdx] +=
                    As[threadRow * TM + resIdx][dotIdx] * Bs[dotIdx][threadCol];
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        const uint rowC = by * BN + threadRow * TM + resIdx;
        const uint colC = bx * BK + threadCol;

        if (rowC < N && colC < K) {
            C[rowC * K + colC] =
                alpha * threadResults[resIdx] + beta * bias[colC];
        }
    }
}

template <const int BN, const int BM, const int BK, const int TM>
__global__ void gemm_kernel_tiled_1D_blocktiling_nt(const float *A,
                                                    const float *B,
                                                    const float *bias, float *C,
                                                    int N, int M, int K,
                                                    float alpha, float beta) {
    __shared__ float As[BN][BM];  // 64, 8
    __shared__ float Bs[BM][BK];  // 8, 64

    // Block coordinates remain the same
    uint bx = blockIdx.x;  // (0 - K div BK)
    uint by = blockIdx.y;  // (0 - N div BN)

    const uint threadCol = threadIdx.x % BK;
    const uint threadRow = threadIdx.x / BK;

    assert(BN * BM == blockDim.x);
    assert(BM * BK == blockDim.x);
    const uint innerColA = threadIdx.x % BM;
    const uint innerRowA = threadIdx.x / BM;
    const uint innerColB = threadIdx.x % BK;
    const uint innerRowB = threadIdx.x / BK;

    // A's loading pattern remains the same
    const uint A_baseoffset = by * BN * M;

    // B's base offset changes since B is transposed
    // Instead of columns, we're now skipping rows in the transposed matrix
    const uint B_baseoffset =
        bx * BK * M;  // Note: multiplied by M instead of 1

    float threadResults[TM] = {0.0};

    for (uint blockTileOffset = 0; blockTileOffset < M; blockTileOffset += BM) {
        // A loading remains identical
        const bool validRowA = (by * BN + innerRowA) < N;
        const bool validColA = (blockTileOffset + innerColA) < M;
        const uint A_idx =
            A_baseoffset + blockTileOffset + innerRowA * M + innerColA;
        if (validRowA && validColA && A_idx < N * M) {
            As[innerRowA][innerColA] = A[A_idx];
        } else {
            As[innerRowA][innerColA] = 0;
        }

        // B loading pattern changes for transposed case
        const bool validRowB = (blockTileOffset + innerRowB) < M;
        const bool validColB = (bx * BK + innerColB) < K;
        // Key change: For transposed B, we access it as B[k][m] instead of
        // B[m][k]
        const uint B_idx =
            B_baseoffset + blockTileOffset + innerColB * M + innerRowB;
        if (validRowB && validColB && B_idx < M * K) {
            Bs[innerRowB][innerColB] = B[B_idx];
        } else {
            Bs[innerRowB][innerColB] = 0;
        }

        __syncthreads();

        // The multiplication logic remains identical since we've loaded
        // the data into shared memory in the same format
        for (uint resIdx = 0; resIdx < TM; ++resIdx) {
            for (uint dotIdx = 0; dotIdx < BM; ++dotIdx) {
                threadResults[resIdx] +=
                    As[threadRow * TM + resIdx][dotIdx] * Bs[dotIdx][threadCol];
            }
        }
        __syncthreads();
    }

    // Result writing remains the same
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        const uint rowC = by * BN + threadRow * TM + resIdx;
        const uint colC = bx * BK + threadCol;

        if (rowC < N && colC < K) {
            C[rowC * K + colC] =
                alpha * threadResults[resIdx] + beta * bias[colC];
        }
    }
}

void gemm_cuda_tiled(const float *A, const float *B, const float *bias,
                     float *out, int n, int m, int k, bool transA, bool transB,
                     float alpha, float beta) {
    const uint blockSize{16};
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim(CEIL_DIV(k, blockSize), CEIL_DIV(n, blockSize));

    gemm_kernel_tiled<blockSize><<<gridDim, blockDim>>>(
        A, B, bias, out, n, m, k, transA, transB, alpha, beta);
}

void gemm_cuda_tiled_1D(const float *A, const float *B, const float *bias,
                        float *out, int n, int m, int k, bool transA,
                        bool transB, float alpha, float beta) {
    const uint blockSize{16};
    dim3 blockDim(blockSize * blockSize);
    dim3 gridDim(CEIL_DIV(k, blockSize), CEIL_DIV(n, blockSize));

    gemm_kernel_tiled_1D<blockSize><<<gridDim, blockDim>>>(
        A, B, bias, out, n, m, k, transA, transB, alpha, beta);
}

void gemm_tiled_1D_blocktiling(const float *A, const float *B,
                               const float *bias, float *out, int n, int m,
                               int k, bool transA, bool transB, float alpha,
                               float beta) {
    const uint BN{64};
    const uint BM{8};
    const uint BK{64};
    const uint TM{8};
    dim3 gridDim(CEIL_DIV(k, BK), CEIL_DIV(n, BN));
    dim3 blockDim((BN * BK) / TM);
    if (!transA && !transB) {
        gemm_kernel_tiled_1D_blocktiling_nn<BN, BM, BK, TM>
            <<<gridDim, blockDim>>>(A, B, bias, out, n, m, k, alpha, beta);
    } else if (!transA && transB) {
        gemm_kernel_tiled_1D_blocktiling_nt<BN, BM, BK, TM>
            <<<gridDim, blockDim>>>(A, B, bias, out, n, m, k, alpha, beta);
    }

    else {
        throw std::runtime_error(
            "Only NN, NT blocktiling kernels are implemented.");
    }
}

void gemm_cuda_naive(const float *A, const float *B, const float *bias,
                     float *out, int n, int m, int k, bool transA, bool transB,
                     float alpha, float beta) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim((k + blockDim.x - 1) / blockDim.x,
                 (n + blockDim.y - 1) / blockDim.y);

    gemm_kernel_naive<<<gridDim, blockDim>>>(A, B, bias, out, n, m, k, transA,
                                             transB, alpha, beta);
}

void gemm_cuda_unoptimized(const float *A, const float *B, const float *bias,
                           float *out, int n, int m, int k, bool transA,
                           bool transB, float alpha, float beta) {
    float *d_A, *d_B, *d_bias, *d_out;

    cudaMalloc((void **)&d_A, n * m * sizeof(float));
    cudaMalloc((void **)&d_B, m * k * sizeof(float));
    cudaMalloc((void **)&d_bias, k * sizeof(float));
    cudaMalloc((void **)&d_out, n * k * sizeof(float));

    cudaMemcpy(d_A, A, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x,
                  (n + blockSize.y - 1) / blockSize.y);

    gemm_kernel_naive<<<gridSize, blockSize>>>(d_A, d_B, d_bias, d_out, n, m, k,
                                               transA, transB, alpha, beta);

    cudaMemcpy(out, d_out, n * k * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_bias);
    cudaFree(d_out);
}

__global__ void relu_kernel(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] < 0 ? 0 : in[idx];
    }
}

void relu_cuda(const float *in, float *out, int n) {
    relu_kernel<<<ceil(n / 32.0), 32>>>(in, out, n);
}

void relu_cuda_unoptimized(const float *in, float *out, int n) {
    float *d_in, *d_out;
    cudaMalloc((void **)&d_in, n * sizeof(float));
    cudaMalloc((void **)&d_out, n * sizeof(float));

    cudaMemcpy(d_in, in, n * sizeof(float), cudaMemcpyHostToDevice);

    relu_kernel<<<ceil(n / 32.0), 32>>>(d_in, d_out, n);

    cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

__global__ void add_kernel(const float *A, const float *B, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = A[idx] + B[idx];
    }
}

void add_cuda(const float *A, const float *B, float *out, int n) {
    add_kernel<<<ceil(n / 32.0), 32>>>(A, B, out, n);
}

void add_cuda_unoptimized(const float *A, const float *B, float *out, int n) {
    float *d_A, *d_B, *d_out;
    cudaMalloc((void **)&d_A, n * sizeof(float));
    cudaMalloc((void **)&d_B, n * sizeof(float));
    cudaMalloc((void **)&d_out, n * sizeof(float));

    cudaMemcpy(d_A, A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * sizeof(float), cudaMemcpyHostToDevice);

    add_kernel<<<ceil(n / 32.0), 32>>>(d_A, d_B, d_out, n);

    cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);
}