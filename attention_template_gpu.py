# -*- coding: utf-8 -*-

"""
Assignment 3 - Attention Kernel Optimisation
GPU Template  (NVIDIA CUDA)
"""

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer
import sys

from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define TILE_DIM 16
#define BLOCK_SIZE 256

// ============================================================================
// METHOD 1: NAIVE -- THREE SEPARATE KERNELS
// ============================================================================

__global__ void qk_tiled_matmul_kernel(const float* Q, const float* K, float* P,
                                        int B, int H, int S, int D) {
    __shared__ float Q_shared[TILE_DIM][TILE_DIM];
    __shared__ float K_shared[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int bh  = blockIdx.z;

    const float* Q_slice = Q + bh * S * D;
    const float* K_slice = K + bh * S * D;
    float*       P_slice = P + bh * S * S;

    float sum = 0.0f;
    for (int t = 0; t < (D + TILE_DIM - 1) / TILE_DIM; ++t) {
        Q_shared[threadIdx.y][threadIdx.x] =
            (row < S && t * TILE_DIM + threadIdx.x < D)
            ? Q_slice[row * D + t * TILE_DIM + threadIdx.x] : 0.0f;
        K_shared[threadIdx.y][threadIdx.x] =
            (col < S && t * TILE_DIM + threadIdx.y < D)
            ? K_slice[col * D + t * TILE_DIM + threadIdx.y] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_DIM; ++k)
            sum += Q_shared[threadIdx.y][k] * K_shared[k][threadIdx.x];
        __syncthreads();
    }
    if (row < S && col < S)
        P_slice[row * S + col] = sum / sqrtf((float)D);
}

__global__ void softmax_kernel(float* P, float* A, int S) {
    __shared__ float s_mem[BLOCK_SIZE];
    int tx  = threadIdx.x;
    int row = blockIdx.x;

    float* row_P = P + row * S;
    float* row_A = A + row * S;

    float local_max = -FLT_MAX;
    for (int i = tx; i < S; i += BLOCK_SIZE)
        if (row_P[i] > local_max) local_max = row_P[i];
    s_mem[tx] = local_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tx < stride && s_mem[tx + stride] > s_mem[tx]) s_mem[tx] = s_mem[tx + stride];
        __syncthreads();
    }
    float row_max = s_mem[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tx; i < S; i += BLOCK_SIZE)
        local_sum += expf(row_P[i] - row_max);
    s_mem[tx] = local_sum;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tx < stride) s_mem[tx] += s_mem[tx + stride];
        __syncthreads();
    }
    float row_sum = s_mem[0];
    __syncthreads();

    for (int i = tx; i < S; i += BLOCK_SIZE)
        row_A[i] = expf(row_P[i] - row_max) / (row_sum + 1e-6f);
}

__global__ void av_matmul_kernel(const float* A, const float* V, float* O,
                                  int B, int H, int S, int D) {
    __shared__ float A_shared[TILE_DIM][TILE_DIM];
    __shared__ float V_shared[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int bh  = blockIdx.z;

    const float* A_slice = A + bh * S * S;
    const float* V_slice = V + bh * S * D;
    float*       O_slice = O + bh * S * D;

    float sum = 0.0f;
    for (int t = 0; t < (S + TILE_DIM - 1) / TILE_DIM; ++t) {
        A_shared[threadIdx.y][threadIdx.x] =
            (row < S && t * TILE_DIM + threadIdx.x < S)
            ? A_slice[row * S + t * TILE_DIM + threadIdx.x] : 0.0f;
        int v_row = t * TILE_DIM + threadIdx.y;
        V_shared[threadIdx.y][threadIdx.x] =
            (v_row < S && col < D)
            ? V_slice[v_row * D + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_DIM; ++k)
            sum += A_shared[threadIdx.y][k] * V_shared[k][threadIdx.x];
        __syncthreads();
    }
    if (row < S && col < D)
        O_slice[row * D + col] = sum;
}

// ============================================================================
// METHOD 2: OPTIMIZED -- TILED QKt + STATS + TILED FUSED SOFTMAX-AV
//
// Removes the S*S A-matrix DRAM roundtrip of the naive path:
// ============================================================================

// Per-row max and sum of exp. row_stats holds 2 values per global row:
// row_stats[2*row + 0] = max, row_stats[2*row + 1] = sum
__global__ void softmax_stats_kernel(const float* P, float* row_stats, int S) {
    __shared__ float s_mem[BLOCK_SIZE];
    int tx  = threadIdx.x;
    int row = blockIdx.x;           // global row index (bh*S + s)

    const float* row_P = P + row * S;

    float local_max = -FLT_MAX;
    for (int i = tx; i < S; i += BLOCK_SIZE)
        if (row_P[i] > local_max) local_max = row_P[i];
    s_mem[tx] = local_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tx < stride && s_mem[tx + stride] > s_mem[tx]) s_mem[tx] = s_mem[tx + stride];
        __syncthreads();
    }
    float row_max = s_mem[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tx; i < S; i += BLOCK_SIZE)
        local_sum += expf(row_P[i] - row_max);
    s_mem[tx] = local_sum;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tx < stride) s_mem[tx] += s_mem[tx + stride];
        __syncthreads();
    }

    if (tx == 0) {
        row_stats[2 * row + 0] = row_max;
        row_stats[2 * row + 1] = s_mem[0] + 1e-6f;
    }
}

// Tiled matmul O = softmax(P) * V.
__global__ void tiled_softmax_av_kernel(const float* P, const float* V,
                                         const float* row_stats, float* O,
                                         int B, int H, int S, int D) {
    __shared__ float A_shared[TILE_DIM][TILE_DIM];  // holds softmax weights
    __shared__ float V_shared[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * blockDim.y + threadIdx.y;  // output row (query index)
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output col (D index)
    int bh  = blockIdx.z;

    const float* P_slice = P + bh * S * S;
    const float* V_slice = V + bh * S * D;
    float*       O_slice = O + bh * S * D;

    float rmax = 0.0f, rsum = 1.0f;
    if (row < S) {
        rmax = row_stats[2 * (bh * S + row) + 0];
        rsum = row_stats[2 * (bh * S + row) + 1];
    }

    float sum = 0.0f;
    for (int t = 0; t < (S + TILE_DIM - 1) / TILE_DIM; ++t) {
        int p_col = t * TILE_DIM + threadIdx.x;
        A_shared[threadIdx.y][threadIdx.x] =
            (row < S && p_col < S)
            ? expf(P_slice[row * S + p_col] - rmax) / rsum : 0.0f;

        int v_row = t * TILE_DIM + threadIdx.y;
        V_shared[threadIdx.y][threadIdx.x] =
            (v_row < S && col < D)
            ? V_slice[v_row * D + col] : 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE_DIM; ++k)
            sum += A_shared[threadIdx.y][k] * V_shared[k][threadIdx.x];
        __syncthreads();
    }
    if (row < S && col < D)
        O_slice[row * D + col] = sum;
}

// ============================================================================
// HOST INTERFACE
// ============================================================================
torch::Tensor attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                 bool use_fused) {
    TORCH_CHECK(Q.device().is_cuda());
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());
    TORCH_CHECK(Q.dtype() == torch::kFloat32);

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    auto O = torch::zeros({B, H, S, D}, Q.options());
    auto P = torch::empty({B, H, S, S}, Q.options());

    dim3 block_mm(TILE_DIM, TILE_DIM);
    dim3 grid_qk((S + TILE_DIM - 1) / TILE_DIM, (S + TILE_DIM - 1) / TILE_DIM, B * H);
    qk_tiled_matmul_kernel<<<grid_qk, block_mm>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), P.data_ptr<float>(), B, H, S, D);

    dim3 grid_av((D + TILE_DIM - 1) / TILE_DIM, (S + TILE_DIM - 1) / TILE_DIM, B * H);

    if (use_fused) {
        auto row_stats = torch::empty({B * H * S * 2}, Q.options());
        softmax_stats_kernel<<<B * H * S, BLOCK_SIZE>>>(
            P.data_ptr<float>(), row_stats.data_ptr<float>(), S);

        tiled_softmax_av_kernel<<<grid_av, block_mm>>>(
            P.data_ptr<float>(), V.data_ptr<float>(),
            row_stats.data_ptr<float>(), O.data_ptr<float>(), B, H, S, D);
    } else {
        auto A = torch::empty({B, H, S, S}, Q.options());
        softmax_kernel<<<B * H * S, BLOCK_SIZE>>>(
            P.data_ptr<float>(), A.data_ptr<float>(), S);
        av_matmul_kernel<<<grid_av, block_mm>>>(
            A.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), B, H, S, D);
    }

    return O;
}
"""

_CPP_DECL = (
    "torch::Tensor attention_forward(torch::Tensor, torch::Tensor, torch::Tensor, bool);"
)

_attn_ext = load_inline(
    name="attn_ext_v10",
    cpp_sources=_CPP_DECL,
    cuda_sources=_CUDA_SRC,
    functions=["attention_forward"],
    extra_cuda_cflags=["-O2", "-Xcompiler", "/Zc:preprocessor"],
    verbose=False,
)

USE_FUSED = True

def attention_cuda(Q, K, V):
    return _attn_ext.attention_forward(Q.contiguous(), K.contiguous(), V.contiguous(), USE_FUSED)

def attention_pytorch(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V)


def check_correctness():
    print("=" * 55)
    print(f"  Step 1: Correctness check  (fused={USE_FUSED})")
    print("=" * 55)

    torch.manual_seed(42)
    configs = [
        (1, 1, 64,  64),
        (2, 4, 128, 64),
        (2, 8, 512, 64),
    ]

    all_passed = True
    for B, H, S, D in configs:
        Q = torch.randn(B, H, S, D, device="cuda")
        K = torch.randn(B, H, S, D, device="cuda")
        V = torch.randn(B, H, S, D, device="cuda")

        out_cuda    = attention_cuda(Q, K, V)
        out_pytorch = attention_pytorch(Q, K, V)

        max_err = (out_cuda - out_pytorch).abs().max().item()
        passed  = max_err < 1e-3
        status  = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"  B={B} H={H} S={S:<4} D={D}  max_err={max_err:.2e}  {status}")

    print()
    if all_passed:
        print("  All configurations passed. Proceed to Step 2.")
    else:
        print("  One or more configurations FAILED.")
    print()
    return all_passed


def run_benchmark():
    print("=" * 55)
    print(f"  Step 2: Benchmark  (fused={USE_FUSED})")
    print("=" * 55)

    B, H, D = 2, 8, 64
    for S in [128, 512, 1024]:
        Q = torch.randn(B, H, S, D, device="cuda")
        K = torch.randn(B, H, S, D, device="cuda")
        V = torch.randn(B, H, S, D, device="cuda")

        for _ in range(5):
            attention_cuda(Q, K, V)
        torch.cuda.synchronize()

        t_cuda = Timer(
            stmt="attention_cuda(Q, K, V)",
            globals={"attention_cuda": attention_cuda, "Q": Q, "K": K, "V": V},
        ).timeit(100)

        t_py = Timer(
            stmt="attention_pytorch(Q, K, V)",
            globals={"attention_pytorch": attention_pytorch, "Q": Q, "K": K, "V": V},
        ).timeit(100)

        slowdown = t_cuda.mean / t_py.mean
        print(f"  S={S:<4} | Custom: {t_cuda.mean*1e3:.3f} ms | PyTorch: {t_py.mean*1e3:.3f} ms | slowdown: {slowdown:.1f}x")
        sys.stdout.flush()

    print()


if __name__ == "__main__":
    passed = check_correctness()
    if passed:
        print("Running benchmark...")
        run_benchmark()
