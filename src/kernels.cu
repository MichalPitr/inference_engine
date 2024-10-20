#include <cuda_runtime.h>

#include <iostream>

#define BLOCK_SIZE 16
#define TILE_SIZE 16

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

    1. Tiles are not (BLOCKSIZE, BLOCKSIZE), but rather (BN, BM) and (BM,
   BK). Important is that each thread has to process (BN*BK) / ((BN*BM +
   BM+BK)/2) = (2*BN*BK)/(BN*BM + BM+BK) results. So then supposing we use
   (64, 8), (8, 64), we get 2*64*64 / (16*64) = 8.

    2. Each thread still only loads one value, but then computes multiple
   outputs. The computation of outputs is done in a way to avoid redundant
   shared memory loads. In practice, this is a mostly pedagogical exercise
   as the compiler will optimize redundant loads away.

    Tensors are of shape:
        A: (n, m)
        B: (m, k)
        C: (n, k)
*/
template <const int BN, const int BM, const int BK, const int TM>
__global__ void gemm_kernel_tiled_1D_blocktiling(const float *A, const float *B,
                                                 const float *bias, float *out,
                                                 int N, int M, int K,
                                                 bool transA, bool transB,
                                                 float alpha, float beta) {
    __shared__ float As[BN][BM];  // 64, 8
    __shared__ float Bs[BM][BK];  // 8, 64

    // Block x y coordinates
    uint bx = blockIdx.x;  // (0 - K div BK)
    uint by = blockIdx.y;  // (0 - N div BN)

    // thread x y coordinates. Coordinates within shared memory block.
    uint tx = threadIdx.x % BK;  // (0 - 63)
    uint ty = threadIdx.x / BK;  // (0 - 7)

    float threadResults[TM] = {0.0};
    // printf("expected num of iterations = %d\n", CEIL_DIV(M, BM));
    for (uint blkIdx = 0; blkIdx < CEIL_DIV(M, BM); ++blkIdx) {
        // printf("BlockIndex = %d\n", blkIdx);
        // Collaboratively load tile
        int A_row = (by * BM + tx);
        int A_col = blkIdx * BM + ty;
        int A_idx = A_row * M + A_col;
        // printf("row=%d, col=%d, A_idx=%d\n", A_row, A_col, A_idx);
        if (A_row < N && A_col < M) {
            // printf(
            //     "Storing A[%d] in As[%d][%d], bx=%d, by=%d, tx=%d, ty=%d, "
            //     "blkIdx=%d, M=%d, N=%d, K=%d, BM=%d, BN=%d, BK=%d, TM=%d\n",
            // A_idx, tx, ty, bx, by, tx, ty, blkIdx, M, N, K, BM, BN, BK, TM);
            // FIXME: implement transpose.
            As[tx][ty] = transA ? A[A_idx] : A[A_idx];
        } else {
            // Out of bounds defaults to 0.
            As[tx][ty] = 0.0f;
        }

        int B_col = (bx * BK + tx);
        int B_row = (blkIdx * BM + ty);
        int B_idx = B_row * K + B_col;
        // printf("row=%d, col=%d, B_idx=%d\n", B_row, B_col, B_idx);
        if (B_row < M && B_col < K) {
            Bs[ty][tx] = transB ? B[B_idx] : B[B_idx];
        } else {
            // Out of bounds defaults to 0.
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // FIXME: Likely bug in this multiplication.
        // Dot product for multiple outputs.
        // We have a [64, 8] and [8, 64] matrix As and Bs. Each (512) thread
        // computes 8 results. This means that each thread computes
        for (uint resIdx = 0; resIdx < TM; ++resIdx) {
            for (uint dotIdx = 0; dotIdx < BM; ++dotIdx) {
                // printf("resIdx=%d, dotIdx=%d, tx=%d, ty=%d\n", resIdx,
                // dotIdx,
                //        tx, ty);

                // As[(ty * TM + resIdx) * BK + dotIdx] * Bs[dotIdx * BN
                // + tx];
                threadResults[resIdx] +=
                    As[ty * TM + resIdx][dotIdx] * Bs[dotIdx][tx];
            }
        }
        __syncthreads();
    }

    //     C[(threadRow * TM + resIdx) * N + threadCol] =
    //         alpha * threadResults[resIdx] +
    //         beta * C[(threadRow * TM + resIdx) * N + threadCol];
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        uint row = ty * TM + resIdx;
        uint col = tx;
        if (row < N && col < K) {
            out[(row * K) + col] =
                threadResults[resIdx] * alpha + bias[col] * beta;
        }
    }
    // printf("Done.\n");
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
    gemm_kernel_tiled_1D_blocktiling<BN, BM, BK, TM><<<gridDim, blockDim>>>(
        A, B, bias, out, n, m, k, transA, transB, alpha, beta);
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