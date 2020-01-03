
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define QUERY 0 // Output device query results
#if QUERY == 1
// Helper for device query
void deviceQuery(cudaDeviceProp devProp);
#endif

#define OPTIMIZATION 6
// List of Optimizations:
// -------------------------------------------------------------------------- //
// 0: Initial version, no optimization
// -------------------------------------------------------------------------- //
// 1: Constant memory for masks/weights
// -------------------------------------------------------------------------- //
// (2.0: CPU version of unroll + gemm, see new-forward.h)
// 2: Unroll + shared-memory Matrix multiply
// -------------------------------------------------------------------------- //
// 3: Kernel fusion for unrolling and matrix-multiplication (square tile, primitive version). primitive version means constants are not hard-coded
// -------------------------------------------------------------------------- //
// (4.0: Use rectangular tile (primitive version))
// (4.1: Customize tile width for each layer & get rid of constant memory. 6x16 for layer1, 16x32 for layer2)
// 4: __restrict__ & #pragma unroll & hardcode all known constants.
// Several promising techniques are also implemented during optimization 4 for layer 2:
//    a) double buffering, i.e. use 2x shared memory to pre-fetch the value for next iteration --> no good
//    b) half-precision operation.
//    c) register cache. (but this implementation is not efficient, it's based on a 1x32 stride and each element will need to load 32 X elements)
// In host code, uncomment layer_2_double_buffer/layer_2_half_precision/layer_2_register_cache to see the results
// -------------------------------------------------------------------------- //
// (5.0: Use register cache (primitive version), but there is a better __shfl_sync thread indexing method than this, see layer_1_6x16N_cache in OPTIMIZATION 5)
// 5: Register cache for both W and X & Thread coarsening.
// Different block size is explored, 4x8 OR 2x16. Different coarsening methods is explored:
//    a) Coarsen Nx work by reload into register cache N times
//    b) Coarsen Nx work by load Nx values into register (this may exceed the per thread register No. limit)
// Timing of each version is documented. In host code, uncomment to see the results
// -------------------------------------------------------------------------- //
// (6.0: Use half precision for optimization 5. Also try use shmem for both W and X)
// (6.1: Try convolution version for both layers. Layer 1 can be improved as good as unroll version, but layer 2 doesn't. Double buffering and B direction coarsening are also tried, but no good)
// 6: Final competition version. Unroll version outperforms convolution version
//    a) Use half precision
//    b) Use shared memory for W & register cache for X
//    c) Different shared memory patterns and coarsening patterns are tested
// -------------------------------------------------------------------------- //

// Timing (Op 1, Op 2):
// 0 - 8.9ms, 24.3ms
// 1 - 8.4ms, 23.1ms
// 2 - 25.6ms, 21.2ms
// 3 - 18.5ms, 15.8ms
// 4 - 3.3ms, 6.1ms
// 5 - 2.3ms, 4.0ms
// 6.0 - 2.1ms, 3.8ms
// 6.1 - 2.0ms, 5.4ms
// 6 - 1.7ms, 2.3ms

#if OPTIMIZATION == 0

#define TILE_WIDTH (32)
__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int H_grid = ceil(1.0*H_out/TILE_WIDTH);
    int W_grid = ceil(1.0*W_out/TILE_WIDTH);
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
    int w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;

    if (h < H_out && w < W_out) { // boundary check
        float result = 0;
        for (int c = 0; c < C; c++) { // sum over all input channels
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    result += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
                }
            }
        }
        y4d(b, m, h, w) = result;
    }

#undef y4d
#undef x4d
#undef k4d
}

#elif OPTIMIZATION == 1

#define TILE_WIDTH (32)
#define MAX_CONSTMEM (64 * 1024 / 4 / 4) // in total 64KB constant memory (=16K floats), but we will need at most 16x6x5x5=2400 floats. Use 1/4 constmem is enough. If use full memory, it will report error.
__constant__ float k[MAX_CONSTMEM]; // constant memory size should be known at compile time. we can hard-code it as Mask[M][C][K][K] for layer 1 & 2 but I don' want to. So, just allocate as the full constant memory space.

// Exactly the same kernel as Optimization 0, except k is not in the function arguments
__global__ void forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K) {
// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int H_grid = ceil(1.0*H_out/TILE_WIDTH);
    int W_grid = ceil(1.0*W_out/TILE_WIDTH);
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
    int w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;

    if (h < H_out && w < W_out) { // boundary check
        float result = 0;
        for (int c = 0; c < C; c++) { // sum over all input channels
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    result += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
                }
            }
        }
        y4d(b, m, h, w) = result;
    }

#undef y4d
#undef x4d
#undef k4d
}
#undef MAX_CONSTMEM

#elif OPTIMIZATION == 2

#define MAX_CONSTMEM (64 * 1024 / 4 / 4) // in total 64KB constant memory (=16K floats), but we will need at most 16x6x5x5=2400 floats. Use 1/4 constmem is enough. If use full memory, it will report error.
__constant__ float k[MAX_CONSTMEM]; // constant memory size should be known at compile time. we can hard-code it as Mask[M][C][K][K] for layer 1 & 2 but I don' want to. So, just allocate as the full constant memory space.

#define UNROLL_TILE_WIDTH (28) // Strategy 2, input threads > output threads, we want input block to be 32x32, so output tile width is (32-K+1)x(32-K+1) = 28x28 where K = 5
__global__ void unroll_kernel(float* x_unroll, float* x, const int B, const int M, const int C, const int H, const int W, const int K) {

#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define x_unroll4d(i3, i2, i1, i0) x_unroll[(i3) * (C * K * K * H_out * W_out) + (i2) * (K * K * H_out * W_out) + (i1) * (H_out * W_out) + i0]
#define mem2d(i, j) shmem[(i) * (UNROLL_TILE_WIDTH + K - 1) + (j)]

    extern __shared__ float shmem[];

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int H_grid = ceil(1.0*H_out/UNROLL_TILE_WIDTH);
    int W_grid = ceil(1.0*W_out/UNROLL_TILE_WIDTH);
    int b = blockIdx.x;
    int c = blockIdx.y;
    int h = blockIdx.z / W_grid * UNROLL_TILE_WIDTH + threadIdx.y; // recover original pixel (i,j) location
    int w = blockIdx.z % W_grid * UNROLL_TILE_WIDTH + threadIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Load (since the convolution is upleft corner centered, no index shift is needed)
    if (h < H && w < W)
        mem2d(ty, tx) = x4d(b, c, h, w);

    __syncthreads(); // necessary

    // Write
    if (ty < UNROLL_TILE_WIDTH && tx < UNROLL_TILE_WIDTH &&
        h < H_out && w < W_out) { // two boundary checks: 1. deactivate marginal input threads 2. mapped pixel location should be within output tile H_out x W_out
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                x_unroll4d(b, c, p*K+q, h*W_out+w) = mem2d(ty+p, tx+q);
                // Be careful of indexing x_unroll4d. It's B x C x (K*K) x (H_out*W_out) follow layout BxCxKxKxH_outxW_out. (K*K) and (H_out*W_out) are both linearized.
                // Memory coalescing analysis here: based on the layout, the (p,q)->(K,K) iteration goes vertically, so adjacent threads will coalesce row, good!
            }
        }
    }

#undef x4d
#undef x_unroll4d
#undef mem2d
}

#define GEMM_TILE_WIDTH (32)
__global__ void gemm_kernel(float* y, float* x_unroll, const int B, const int M, const int C, const int H, const int W, const int K) {

    __shared__ float tileW[GEMM_TILE_WIDTH][GEMM_TILE_WIDTH];
    __shared__ float tileX[GEMM_TILE_WIDTH][GEMM_TILE_WIDTH];
    // Ideas of further optimizations:
    // 1. one-thread-multi-element (rectangular shared memory) load & write
    // 2. adjust tile size, right now layer M1 = 6, layer M2 = 16, both much smaller than TILE_WIDTH=32, so significant control divergence and many halo 0s in shmem

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    // use z (batch No.) to offset indexing when loading X & writing Y
    int offset_x = blockIdx.z * C * K * K * H_out * W_out;
    int offset_y = blockIdx.z * M * H_out * W_out;

    float result = 0;
    for (int t = 0; t < ceil(1.0*C*K*K/GEMM_TILE_WIDTH); t++) {
        // Load
        tileW[ty][tx] = Row < M && t*GEMM_TILE_WIDTH+tx < C * K * K ? k[Row*(C*K*K) + (t*GEMM_TILE_WIDTH+tx)]: 0;
        tileX[ty][tx] = t*GEMM_TILE_WIDTH+ty < C * K * K && Col < H_out * W_out ? x_unroll[offset_x + (t*GEMM_TILE_WIDTH+ty)*H_out*W_out + Col] : 0; // Note the b*... offset
        __syncthreads();

        // Compute
        for (int i = 0; i < GEMM_TILE_WIDTH; i++)
            result += tileW[ty][i] * tileX[i][tx];
        __syncthreads();
    }

    // Write
    if (Row < M && Col < H_out * W_out)
        y[offset_y + Row * H_out * W_out + Col] = result;
}

#elif OPTIMIZATION == 3

#define MAX_CONSTMEM (64 * 1024 / 4 / 4) // in total 64KB constant memory (=16K floats), but we will need at most 16x6x5x5=2400 floats. Use 1/4 constmem is enough. If use full memory, it will report error.
__constant__ float k[MAX_CONSTMEM]; // constant memory size should be known at compile time. we can hard-code it as Mask[M][C][K][K] for layer 1 & 2 but I don' want to. So, just allocate as the full constant memory space.

#define TILE_WIDTH (32)
__global__ void fused_kernel(float *y, float *x, const int B, const int M, const int C, const int H, const int W, const int K) {

#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

    __shared__ float tileW[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileX[TILE_WIDTH][TILE_WIDTH];
    // Ideas of further optimizations:
    // 1. one-thread-multi-element (rectangular shared memory) load & write
    // 2. adjust tile size, right now layer M1 = 6, layer M2 = 16, both much smaller than TILE_WIDTH=32, so significant control divergence and many halo 0s in shmem

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    // recover pixel location (h, w) in the input images
    int h = Col / W_out;
    int w = Col % W_out;

    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    float result = 0;
    int Row_x, c, kk;
    for (int t = 0; t < ceil(1.0*C*K*K/TILE_WIDTH); t++) {
        // Load W tile
        tileW[ty][tx] = Row < M && t*TILE_WIDTH+tx < C * K * K ? k[Row*(C*K*K) + (t*TILE_WIDTH+tx)]: 0;

        // Load unrolled X tile
        // (Conceptual) Unrolled X is (B*C*K*K)x(H_out*W_out) matrix, so we should figure out several indices
        Row_x = t*TILE_WIDTH+ty;
        c = Row_x / (K*K); // recover input channel index
        kk = Row_x % (K*K); // recover linearized KxK mask index
        tileX[ty][tx] = Row_x < C * K * K && Col < H_out * W_out ? x4d(b, c, h + kk/K, w + kk%K) : 0;
        __syncthreads(); // no sync needed between tileW & tileX

        // Compute
        for (int i = 0; i < TILE_WIDTH; i++)
            result += tileW[ty][i] * tileX[i][tx];
        __syncthreads();
    }

    // Write
    if (Row < M && Col < H_out * W_out)
        y[b * M * H_out * W_out + Row * H_out * W_out + Col] = result;

#undef x4d
}

#elif OPTIMIZATION == 40

#define MAX_CONSTMEM (64 * 1024 / 4 / 4) // in total 64KB constant memory (=16K floats), but we will need at most 16x6x5x5=2400 floats. Use 1/4 constmem is enough. If use full memory, it will report error.
__constant__ float k[MAX_CONSTMEM]; // constant memory size should be known at compile time. we can hard-code it as Mask[M][C][K][K] for layer 1 & 2 but I don' want to. So, just allocate as the full constant memory space.

// Rectangular tile
#define TILE_WIDTH_Y (16)
#define TILE_WIDTH_X (32) // recall [ty][tx], Y is vertical/row, X is horizontal/col

__global__ void forward_kernel(float *y, float *x, const int B, const int M, const int C, const int H, const int W, const int K) {

#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

    __shared__ float tileW[TILE_WIDTH_Y][TILE_WIDTH_X];
    __shared__ float tileX[TILE_WIDTH_X][TILE_WIDTH_X];

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    // recover pixel location (h, w) in the input images
    int h = Col / W_out;
    int w = Col % W_out;

    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    float result = 0;
    int Row_x, c, kk;
    for (int t = 0; t < ceil(1.0*C*K*K/TILE_WIDTH_X); t++) {
        // Load W tile (as normal)
        tileW[ty][tx] = Row < M && t*TILE_WIDTH_X+tx < C * K * K ? k[Row*(C*K*K) + (t*TILE_WIDTH_X+tx)]: 0;

        // Load unrolled X tile (each thread repeats TILE_WIDTH_X/TILE_WIDTH_Y times of work)
        // (Conceptual) Unrolled X is (B*C*K*K)x(H_out*W_out) matrix, so we should figure out several indices
        for (int i = 0; i < (TILE_WIDTH_X + TILE_WIDTH_Y - 1)/TILE_WIDTH_Y; i++) {
            if (ty+i*TILE_WIDTH_Y < TILE_WIDTH_X) { // boundary check
                Row_x = t*TILE_WIDTH_X + i*TILE_WIDTH_Y + ty;
                c = Row_x / (K*K); // recover input channel index
                kk = Row_x % (K*K); // recover linearized KxK mask index
                tileX[ty+i*TILE_WIDTH_Y][tx] = Row_x < C * K * K && Col < H_out * W_out ? x4d(b, c, h + kk/K, w + kk%K) : 0;
            }
        }
        __syncthreads(); // no sync needed between tileW & tileX

        // Compute
        for (int i = 0; i < TILE_WIDTH_X; i++)
            result += tileW[ty][i] * tileX[i][tx];
        __syncthreads();
    }

    // Write
    if (Row < M && Col < H_out * W_out)
        y[b * M * H_out * W_out + Row * H_out * W_out + Col] = result;

#undef x4d
}

#elif OPTIMIZATION == 41

// Rectangular tile
#define TILE_WIDTH_Y_1 (6)
#define TILE_WIDTH_X_1 (32) // recall [ty][tx], Y is vertical/row, X is horizontal/col

__global__ void layer_1(float *y, float *x, float *k, const int B, const int M, const int C, const int H, const int W, const int K) {

#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

    __shared__ float tileW[TILE_WIDTH_Y_1][TILE_WIDTH_X_1];
    __shared__ float tileX[TILE_WIDTH_X_1][TILE_WIDTH_X_1];

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    // recover pixel location (h, w) in the input images
    int h = Col / W_out;
    int w = Col % W_out;

    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    float result = 0;
    int Row_x, c, kk;
    for (int t = 0; t < ceil(1.0*C*K*K/TILE_WIDTH_X_1); t++) {
        // Load W tile (as normal)
        tileW[ty][tx] = Row < M && t*TILE_WIDTH_X_1+tx < C * K * K ? k[Row*(C*K*K) + (t*TILE_WIDTH_X_1+tx)]: 0;

        // Load unrolled X tile (each thread repeats TILE_WIDTH_X/TILE_WIDTH_Y times of work)
        // (Conceptual) Unrolled X is (B*C*K*K)x(H_out*W_out) matrix, so we should figure out several indices
        for (int i = 0; i < (TILE_WIDTH_X_1 + TILE_WIDTH_Y_1 - 1)/TILE_WIDTH_Y_1; i++) {
            if (ty+i*TILE_WIDTH_Y_1 < TILE_WIDTH_X_1) { // boundary check
                Row_x = t*TILE_WIDTH_X_1 + i*TILE_WIDTH_Y_1 + ty;
                c = Row_x / (K*K); // recover input channel index
                kk = Row_x % (K*K); // recover linearized KxK mask index
                tileX[ty+i*TILE_WIDTH_Y_1][tx] = Row_x < C * K * K && Col < H_out * W_out ? x4d(b, c, h + kk/K, w + kk%K) : 0;
            }
        }
        __syncthreads(); // no sync needed between tileW & tileX

        // Compute
        for (int i = 0; i < TILE_WIDTH_X_1; i++)
            result += tileW[ty][i] * tileX[i][tx];
        __syncthreads();
    }

    // Write
    if (Row < M && Col < H_out * W_out)
        y[b * M * H_out * W_out + Row * H_out * W_out + Col] = result;

#undef x4d
}

// Rectangular tile
#define TILE_WIDTH_Y_2 (16)
#define TILE_WIDTH_X_2 (32) // recall [ty][tx], Y is vertical/row, X is horizontal/col

__global__ void layer_2(float *y, float *x, float *k, const int B, const int M, const int C, const int H, const int W, const int K) {

#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

    __shared__ float tileW[TILE_WIDTH_Y_2][TILE_WIDTH_X_2];
    __shared__ float tileX[TILE_WIDTH_X_2][TILE_WIDTH_X_2];

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    // recover pixel location (h, w) in the input images
    int h = Col / W_out;
    int w = Col % W_out;

    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    float result = 0;
    int Row_x, c, kk;
    for (int t = 0; t < ceil(1.0*C*K*K/TILE_WIDTH_X_2); t++) {
        // Load W tile (as normal)
        tileW[ty][tx] = Row < M && t*TILE_WIDTH_X_2+tx < C * K * K ? k[Row*(C*K*K) + (t*TILE_WIDTH_X_2+tx)]: 0;

        // Load unrolled X tile (each thread repeats TILE_WIDTH_X/TILE_WIDTH_Y times of work)
        // (Conceptual) Unrolled X is (B*C*K*K)x(H_out*W_out) matrix, so we should figure out several indices
        for (int i = 0; i < (TILE_WIDTH_X_2 + TILE_WIDTH_Y_2 - 1)/TILE_WIDTH_Y_2; i++) {
            if (ty+i*TILE_WIDTH_Y_2 < TILE_WIDTH_X_2) { // boundary check
                Row_x = t*TILE_WIDTH_X_2 + i*TILE_WIDTH_Y_2 + ty;
                c = Row_x / (K*K); // recover input channel index
                kk = Row_x % (K*K); // recover linearized KxK mask index
                tileX[ty+i*TILE_WIDTH_Y_2][tx] = Row_x < C * K * K && Col < H_out * W_out ? x4d(b, c, h + kk/K, w + kk%K) : 0;
            }
        }
        __syncthreads(); // no sync needed between tileW & tileX

        // Compute
        for (int i = 0; i < TILE_WIDTH_X_2; i++)
            result += tileW[ty][i] * tileX[i][tx];
        __syncthreads();
    }

    // Write
    if (Row < M && Col < H_out * W_out)
        y[b * M * H_out * W_out + Row * H_out * W_out + Col] = result;

#undef x4d
}

#elif OPTIMIZATION == 4

// Rectangular tile
#define TILE_WIDTH_Y_1 (6)
#define TILE_WIDTH_X_1 (32)

__global__ void layer_1(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {
// Constants
// M = 6, C = 1, K = 5, H = W = 48, H_out = W_out = 44

#define x4d(i3, i2, i1, i0) x[(i3) * 2304 + (i2) * 2304 + (i1) * 48 + i0]

    __shared__ float tileW[TILE_WIDTH_Y_1][TILE_WIDTH_X_1];
    __shared__ float tileX[TILE_WIDTH_X_1][TILE_WIDTH_X_1];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    // recover pixel location (h, w) in the input images
    int h = Col / 44;
    int w = Col % 44;

    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    float result = 0;
    int Row_x, c, kk;

    // No. of Iteration = 1, so directly remove the for loop... and eliminate one __syncthreads

    // Load W tile (as normal)
    tileW[ty][tx] = /*Row < 6 && */tx < 25 ? k[Row*25 + tx]: 0;

    // Load unrolled X tile (each thread repeats TILE_WIDTH_X/TILE_WIDTH_Y times of work)
    // (Conceptual) Unrolled X is (B*C*K*K)x(H_out*W_out) matrix, so we should figure out several indices
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        if (ty+i*TILE_WIDTH_Y_1 < TILE_WIDTH_X_1) { // boundary check
            Row_x = i*TILE_WIDTH_Y_1 + ty;
            c = Row_x / 25;
            kk = Row_x % 25; // recover linearized KxK mask index
            tileX[ty+i*TILE_WIDTH_Y_1][tx] = Row_x < 25 && Col < 1936 ? x4d(b, c, h + kk/5, w + kk%5) : 0;
        }
    }
    __syncthreads(); // no sync needed between tileW & tileX

    // Compute
    #pragma unroll
    for (int i = 0; i < TILE_WIDTH_X_1; i++)
        result += tileW[ty][i] * tileX[i][tx];

    // Write
    if (/*Row < 6 && */Col < 1936)
        y[b * 11616 + Row * 1936 + Col] = result;

#undef x4d
}

// Rectangular tile
#define TILE_WIDTH_Y_2 (16)
#define TILE_WIDTH_X_2 (32) // recall [ty][tx], Y is vertical/row, X is horizontal/col

__global__ void layer_2(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {
// Constants
// M = 16, C = 6, K = 5, H = W = 22, H_out = W_out = 18

#define x4d(i3, i2, i1, i0) x[(i3) * 2904 + (i2) * 484 + (i1) * 22 + i0]

    __shared__ float tileW[TILE_WIDTH_Y_2][TILE_WIDTH_X_2];
    __shared__ float tileX[TILE_WIDTH_X_2][TILE_WIDTH_X_2];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    // recover pixel location (h, w) in the input images
    int h = Col / 18;
    int w = Col % 18;

    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    float result = 0;
    int Row_x, c, kk;
    #pragma unroll
    for (int t = 0; t < 5; t++) {
        // Load W tile (as normal)
        tileW[ty][tx] = /*Row < 16 && */t*TILE_WIDTH_X_2+tx < 150 ? k[Row*150 + (t*TILE_WIDTH_X_2+tx)] : 0;

        // Load unrolled X tile (each thread repeats TILE_WIDTH_X/TILE_WIDTH_Y times of work)
        // (Conceptual) Unrolled X is (B*C*K*K)x(H_out*W_out) matrix, so we should figure out several indices
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            //if (ty+i*TILE_WIDTH_Y_2 < TILE_WIDTH_X_2) { // boundary check
                Row_x = t*TILE_WIDTH_X_2 + i*TILE_WIDTH_Y_2 + ty;
                c = Row_x / 25;
                kk = Row_x % 25; // recover linearized KxK mask index
                tileX[ty+i*TILE_WIDTH_Y_2][tx] = Row_x < 150 && Col < 324 ? x4d(b, c, h + kk/5, w + kk%5) : 0;
            //}
        }
        __syncthreads(); // no sync needed between tileW & tileX

        // Compute
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH_X_2; i++)
            result += tileW[ty][i] * tileX[i][tx];
        __syncthreads();
    }

    // Write
    if (/*Row < 16 && */Col < 324)
        y[b * 5184 + Row * 324 + Col] = result;

#undef x4d
}

__global__ void layer_2_double_buffer(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {
// Constants
// M = 16, C = 6, K = 5, H = W = 22, H_out = W_out = 18

#define x4d(i3, i2, i1, i0) x[(i3) * 2904 + (i2) * 484 + (i1) * 22 + i0]

    __shared__ float tileW0[TILE_WIDTH_Y_2][TILE_WIDTH_X_2];
    __shared__ float tileX0[TILE_WIDTH_X_2][TILE_WIDTH_X_2];
    __shared__ float tileW1[TILE_WIDTH_Y_2][TILE_WIDTH_X_2];
    __shared__ float tileX1[TILE_WIDTH_X_2][TILE_WIDTH_X_2];

    // double buffering
    float (*tileW_src)[TILE_WIDTH_X_2] = tileW0;
    float (*tileW_dst)[TILE_WIDTH_X_2] = tileW1;
    float (*tileX_src)[TILE_WIDTH_X_2] = tileX0;
    float (*tileX_dst)[TILE_WIDTH_X_2] = tileX1;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    // recover pixel location (h, w) in the input images
    int h = Col / 18;
    int w = Col % 18;

    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    float result = 0;
    int Row_x, c, kk;

    // t = 0
    // Load W tile (as normal)
    tileW_src[ty][tx] = Row < 16 && tx < 150 ? k[Row*150 + tx]: 0;

    // Load unrolled X tile (each thread repeats TILE_WIDTH_X/TILE_WIDTH_Y times of work)
    // (Conceptual) Unrolled X is (B*C*K*K)x(H_out*W_out) matrix, so we should figure out several indices
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        if (ty+i*TILE_WIDTH_Y_2 < TILE_WIDTH_X_2) { // boundary check
            Row_x = i*TILE_WIDTH_Y_2 + ty;
            c = Row_x / 25; // recover input channel index
            kk = Row_x % 25; // recover linearized KxK mask index
            tileX_src[ty+i*TILE_WIDTH_Y_2][tx] = Row_x < 150 && Col < 324 ? x4d(b, c, h + kk/5, w + kk%5) : 0;
        }
    }

    float (*temp_W)[TILE_WIDTH_X_2], (*temp_X)[TILE_WIDTH_X_2];
    #pragma unroll
    for (int t = 0; t < 5; t++) {
        __syncthreads(); // no sync needed between tileW & tileX

        // Compute
        for (int i = 0; i < TILE_WIDTH_X_2; i++)
            result += tileW_src[ty][i] * tileX_src[i][tx];

        // Load W tile (as normal)
        tileW_dst[ty][tx] = Row < 16 && (t+1)*TILE_WIDTH_X_2+tx < 150 ? k[Row*150 + ((t+1)*TILE_WIDTH_X_2+tx)]: 0;

        // Load unrolled X tile (each thread repeats TILE_WIDTH_X/TILE_WIDTH_Y times of work)
        // (Conceptual) Unrolled X is (B*C*K*K)x(H_out*W_out) matrix, so we should figure out several indices
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            if (ty+i*TILE_WIDTH_Y_2 < TILE_WIDTH_X_2) { // boundary check
                Row_x = (t+1)*TILE_WIDTH_X_2 + i*TILE_WIDTH_Y_2 + ty;
                c = Row_x / 25; // recover input channel index
                kk = Row_x % 25; // recover linearized KxK mask index
                tileX_dst[ty+i*TILE_WIDTH_Y_2][tx] = Row_x < 150 && Col < 324 ? x4d(b, c, h + kk/5, w + kk%5) : 0;
            }
        }

        // Swap
        temp_W = tileW_dst;
        tileW_dst = tileW_src;
        tileW_src = temp_W;
        temp_X = tileX_dst;
        tileX_dst = tileX_src;
        tileX_src = temp_X;
    }

    // Write
    if (Row < 16 && Col < 324)
        y[b * 5184 + Row * 324 + Col] = result;

#undef x4d
}

__global__ void layer_2_half_precision(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {
// Constants
// M = 16, C = 6, K = 5, H = W = 22, H_out = W_out = 18

#define x4d(i3, i2, i1, i0) x[(i3) * 2904 + (i2) * 484 + (i1) * 22 + i0]

    __shared__ float tileW[TILE_WIDTH_Y_2][TILE_WIDTH_X_2];
    __shared__ float tileX[TILE_WIDTH_X_2][TILE_WIDTH_X_2];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    // recover pixel location (h, w) in the input images
    int h = Col / 18;
    int w = Col % 18;

    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    half2 result = __float2half2_rn(0);
    int Row_x, kk;
    #pragma unroll
    for (int t = 0; t < 5; t++) {
        // Load W tile (as normal)
        tileW[ty][tx] = /*Row < 16 && */t*TILE_WIDTH_X_2+tx < 150 ? k[Row*150 + (t*TILE_WIDTH_X_2+tx)] : 0;

        // Load unrolled X tile (each thread repeats TILE_WIDTH_X/TILE_WIDTH_Y times of work)
        // (Conceptual) Unrolled X is (B*C*K*K)x(H_out*W_out) matrix, so we should figure out several indices
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            //if (ty+i*TILE_WIDTH_Y_2 < TILE_WIDTH_X_2) { // boundary check
                Row_x = t*TILE_WIDTH_X_2 + i*TILE_WIDTH_Y_2 + ty;
                kk = Row_x % 25; // recover linearized KxK mask index
                tileX[ty+i*TILE_WIDTH_Y_2][tx] = Row_x < 150 && Col < 324 ? x4d(b, Row_x / 25, h + kk/5, w + kk%5) : 0;
            //}
        }
        __syncthreads(); // no sync needed between tileW & tileX

        // Compute
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH_X_2 / 2; i++)
            result = __hfma2(__floats2half2_rn(tileW[ty][2*i], tileW[ty][2*i+1]), __floats2half2_rn(tileX[2*i][tx], tileX[2*i+1][tx]), result);
        __syncthreads();
    }

    // Write
    if (/*Row < 16 && */Col < 324)
        y[b * 5184 + Row * 324 + Col] = __high2float(result) + __low2float(result);

#undef x4d
}

__global__ void layer_2_register_cache(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {

#define x4d(i3, i2, i1, i0) x[(i3) * 2904 + (i2) * 484 + (i1) * 22 + i0]

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    // recover pixel location (h, w) in the input images
    int h = Col / 18;
    int w = Col % 18;

    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    float rc[TILE_WIDTH_X_2]; // a vertical line in tileX, private
    float rc_shared; // a row element in tileW, shared

    float result = 0;
    int Row_x, c, kk;
    #pragma unroll
    for (int t = 0; t < 5; t++) {
        // Load W tile
        rc_shared = /*Row < 16 && */t*TILE_WIDTH_X_2+tx < 150 ? k[Row*150 + (t*TILE_WIDTH_X_2+tx)]: 0;

        // Load unrolled X tile
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH_X_2; i++) {
            Row_x = t*TILE_WIDTH_X_2 + i;
            c = Row_x / 25; // recover input channel index
            kk = Row_x % 25; // recover linearized KxK mask index
            rc[i] = Row_x < 150 && Col < 324 ? x4d(b, c, h + kk/5, w + kk%5) : 0;
        }

        // register cache, no sync needed

        // Compute
        unsigned mask;
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH_X_2; i++) {
            mask = __activemask();
            result += __shfl_sync(mask, rc_shared, i) * rc[i];
        }
    }

    // Write
    if (/*Row < 16 && */Col < 324)
        y[b * 5184 + Row * 324 + Col] = result;

#undef x4d
}


#elif OPTIMIZATION == 50

// Thinking:
// warp-level shared, so maybe make block/tile as 4x8 or 2x16 so that within that block
// all calculations can be register shared? For example, I can still use 4x8 to do work as 16x32, each thread 4x work
// in this way no sync is needed at all, all benefitted from SIMD

// Rectangular tile
#define TILE_WIDTH_Y (4)
#define TILE_WIDTH_X (8)
#define TILE_LOAD_Y (4)
#define TILE_LOAD_X (2)
// each block is as small as 4x8, but it will do 4x work in Y dir and 2x work in X dir, being equivalent to a 16x16 square tile
// why 4x8? 4x8=32 is exactly the warp size. And for layer 2, M = 16 is a multiple of 4
#define WARP_SIZE    (32)

__global__ void forward_kernel(float *y, float *x, float *k, const int B, const int M, const int C, const int H, const int W, const int K) {

#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row_base = blockIdx.y * (TILE_LOAD_Y * blockDim.y) + threadIdx.y;
    int Col_base = blockIdx.x * (TILE_LOAD_X * blockDim.x) + threadIdx.x;
    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    // rc for register cache
    float rc_W[TILE_LOAD_Y * TILE_LOAD_X];
    float rc_X[TILE_LOAD_Y * TILE_LOAD_X];
    float rc_result[TILE_LOAD_Y * TILE_LOAD_X] = { 0 }; // zero out

    int m, Row, Col, Row_x, c, kk, h, w;
    for (int t = 0; t < ceil(1.0*C*K*K/(TILE_LOAD_X*TILE_WIDTH_X)); t++) {
        m = t * TILE_LOAD_X * TILE_WIDTH_X; // Col in W & Row in X

        // Load tileW & tileX into register cache
        for (int i = 0; i < TILE_LOAD_Y; i++) {
            Row = Row_base + i * blockDim.y; // coarsened row index
            Row_x = m + i * TILE_WIDTH_Y + ty; // same for each j iteration
            c = Row_x / (K*K); // recover input channel index
            kk = Row_x % (K*K); // recover linearized KxK mask index
            for (int j = 0; j < TILE_LOAD_X; j++) {
                // Load W
                rc_W[i * TILE_LOAD_X + j] = Row < M && m + j * TILE_WIDTH_X + tx < C * K * K ? k[Row*(C*K*K) + (m + j * TILE_WIDTH_X + tx)] : 0;
                // Load X
                Col = Col_base + j * blockDim.x;
                h = Col / W_out; // recover pixel location (h, w) in the input images
                w = Col % W_out;
                rc_X[i * TILE_LOAD_X + j] = Row_x < C * K * K && Col < H_out * W_out ? x4d(b, c, h + kk/K, w + kk%K) : 0;
            }
        }

        // register cache, no sync needed

        // Compute & Accumulate result of each tile (with warp-level shared)
        for (int i = 0; i < TILE_LOAD_Y; i++) { // when i = 0, access rc_W[0/1]; i = 1, access rc_W[2/3]; i = 2, access rc_W[4/5]; i = 3, access rc_W[6/7];
            for (int j = 0; j < TILE_LOAD_X; j++) { // when j = 0, access rc_X[0/2/4/6]; j = 1, access rc_X[1/3/5/7]
                for (int n = 0; n < TILE_WIDTH_Y; n++) {
                    unsigned mask = __activemask(); // 0xffffffff;
                    rc_result[i * TILE_LOAD_X + j] += __shfl_sync(mask, rc_W[2*i], ty * TILE_WIDTH_X + (ty + n) % TILE_WIDTH_Y)                * __shfl_sync(mask, rc_X[j], (ty + n) % TILE_WIDTH_Y * TILE_WIDTH_X + tx);
                    rc_result[i * TILE_LOAD_X + j] += __shfl_sync(mask, rc_W[2*i], ty * TILE_WIDTH_X + (ty + n) % TILE_WIDTH_Y + TILE_WIDTH_Y) * __shfl_sync(mask, rc_X[j+2], (ty + n) % TILE_WIDTH_Y * TILE_WIDTH_X + tx);
                    rc_result[i * TILE_LOAD_X + j] += __shfl_sync(mask, rc_W[2*i+1], ty * TILE_WIDTH_X + (ty + n) % TILE_WIDTH_Y)                * __shfl_sync(mask, rc_X[j+4], (ty + n) % TILE_WIDTH_Y * TILE_WIDTH_X + tx);
                    rc_result[i * TILE_LOAD_X + j] += __shfl_sync(mask, rc_W[2*i+1], ty * TILE_WIDTH_X + (ty + n) % TILE_WIDTH_Y + TILE_WIDTH_Y) * __shfl_sync(mask, rc_X[j+6], (ty + n) % TILE_WIDTH_Y * TILE_WIDTH_X + tx);
                }
            }
        }

    }

    // Write
    for (int i = 0; i < TILE_LOAD_Y; i++) {
        Row = Row_base + i * blockDim.y;
        for (int j = 0; j < TILE_LOAD_X; j++) {
            Col = Col_base + j * blockDim.x;
            if (Row < M && Col < H_out * W_out)
                y[b * M * H_out * W_out + Row * H_out * W_out + Col] = rc_result[i * TILE_LOAD_X + j];
        }
    }

#undef x4d
}

#elif OPTIMIZATION == 5

// ------------------------------- Layer 1 ---------------------------------- //
// Version 0 (4.1 ms): 4x8 block, 16x16 output tile, 16x16 W tile, 16x16 X tile. Same as layer 2
#define TILE_WIDTH_Y_1 (4)
#define TILE_WIDTH_X_1 (8)
#define TILE_LOAD_Y_1 (4)
#define TILE_LOAD_X_1 (2)
// each block is as small as 4x8, but it will do 4x work in Y dir and 2x work in X dir, being equivalent to a 16x16 square tile
// why 4x8? 4x8=32 is exactly the warp size. And for layer 2, M = 16 is a multiple of 4
__global__ void layer_1_16x16(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {

#define x4d(i3, i2, i1, i0) x[(i3) * 2304 + (i2) * 2304 + (i1) * 48 + i0]

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row_base = blockIdx.y * (TILE_LOAD_Y_1 * blockDim.y) + threadIdx.y;
    int Col_base = blockIdx.x * (TILE_LOAD_X_1 * blockDim.x) + threadIdx.x;
    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    // rc for register cache
    float rc_W[TILE_LOAD_Y_1 * TILE_LOAD_X_1];
    float rc_X[TILE_LOAD_Y_1 * TILE_LOAD_X_1];
    float rc_result[TILE_LOAD_Y_1 * TILE_LOAD_X_1] = { 0 }; // zero out

    int m, Row, Col, Row_x, kk, h, w, id, id_y, id_x, i, j, t, n;
    unsigned mask = 0xffffffff; // __activemask();

    #pragma unroll
    for (t = 0; t < 2; t++) {
        m = t * TILE_LOAD_X_1 * TILE_WIDTH_X_1; // Col in W & Row in X

        // Load tileW & tileX into register cache
        #pragma unroll
        for (i = 0; i < TILE_LOAD_Y_1; i++) {
            Row = Row_base + i * blockDim.y; // coarsened row index
            Row_x = m + i * TILE_WIDTH_Y_1 + ty; // same for each j iteration
            kk = Row_x % 25; // recover linearized KxK mask index
            #pragma unroll
            for (j = 0; j < TILE_LOAD_X_1; j++) {
                id = i * TILE_LOAD_X_1 + j;
                // Load W
                if (Row < 6 && m + j * TILE_WIDTH_X_1 + tx < 25)
                    rc_W[id] = k[Row*25 + (m + j * TILE_WIDTH_X_1 + tx)];
                else
                    rc_W[id] = 0.0;

                // Load X
                Col = Col_base + j * blockDim.x;
                h = Col / 44; // recover pixel location (h, w) in the input images
                w = Col % 44;
                if (Row_x < 25 && Col < 1936)
                    rc_X[id] = x4d(b, Row_x / 25, h + kk/5, w + kk%5);
                else
                    rc_X[id] = 0.0;

            }
        }

        // register cache, no sync needed

        // Compute & Accumulate result of each tile (with warp-level shared)
        #pragma unroll
        for (i = 0; i < TILE_LOAD_Y_1; i++) { // when i = 0, access rc_W[0/1]; i = 1, access rc_W[2/3]; i = 2, access rc_W[4/5]; i = 3, access rc_W[6/7];
            #pragma unroll
            for (j = 0; j < TILE_LOAD_X_1; j++) { // when j = 0, access rc_X[0/2/4/6]; j = 1, access rc_X[1/3/5/7]
                id = i * TILE_LOAD_X_1 + j;
                #pragma unroll
                for (n = 0; n < TILE_WIDTH_Y_1; n++) {
                    id_y = ty * TILE_WIDTH_X_1 + (ty + n) % TILE_WIDTH_Y_1;
                    id_x = (ty + n) % TILE_WIDTH_Y_1 * TILE_WIDTH_X_1 + tx;
                    // 4x8 block, 16x16 tile version:
                    rc_result[id] += __shfl_sync(mask, rc_W[2*i], id_y)                * __shfl_sync(mask, rc_X[j], id_x);
                    rc_result[id] += __shfl_sync(mask, rc_W[2*i], id_y + TILE_WIDTH_Y_1) * __shfl_sync(mask, rc_X[j+2], id_x);
                    rc_result[id] += __shfl_sync(mask, rc_W[2*i+1], id_y)                * __shfl_sync(mask, rc_X[j+4], id_x);
                    rc_result[id] += __shfl_sync(mask, rc_W[2*i+1], id_y + TILE_WIDTH_Y_1) * __shfl_sync(mask, rc_X[j+6], id_x);
                }
            }
        }

    }

    // Write
    #pragma unroll
    for (i = 0; i < TILE_LOAD_Y_1; i++) {
        Row = Row_base + i * blockDim.y;
        #pragma unroll
        for (j = 0; j < TILE_LOAD_X_1; j++) {
            Col = Col_base + j * blockDim.x;
            if (Row < 6 && Col < 1936)
                y[b * 11616 + Row * 1936 + Col] = rc_result[i * TILE_LOAD_X_1 + j];
        }
    }

#undef x4d
}

// Version 1 (2.7ms): 2x16 block, 6x16 output tile, 6x16 W tile, 16x16 X tile
__global__ void layer_1_6x16(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {

#define x4d(i3, i2, i1, i0) x[(i3) * 2304 + (i2) * 2304 + (i1) * 48 + i0]

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row_base = blockIdx.y * (3 * blockDim.y) + threadIdx.y;
    int Col_base = blockIdx.x * (1 * blockDim.x) + threadIdx.x;
    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    // rc for register cache
    float rc_W[3];
    float rc_X[8];
    float rc_result[3] = { 0 }; // zero out

    int m, Row, Col, Row_x, c, kk, h, w, i, j, t;
    unsigned mask = 0xffffffff; // __activemask();

    #pragma unroll
    for (t = 0; t < 2; t++) {
        m = t * 16; // Col in W & Row in X

        // Load tileW & tileX into register cache
        // W
        #pragma unroll
        for (i = 0; i < 3; i++) {
            Row = Row_base + i * blockDim.y;
            if (Row < 6 && m + tx < 25)
                rc_W[i] = k[Row*25 + (m + tx)];
            else
                rc_W[i] = 0.0;
        }
        // X
        #pragma unroll
        for (i = 0; i < 8; i++) {
            Row_x = m + i * blockDim.y + ty;
            c = Row_x / 25;
            kk = Row_x % 25;
            Col = Col_base;
            h = Col / 44;
            w = Col % 44;
            if (Row_x < 25 && Col < 1936)
                rc_X[i] = x4d(b, c, h + kk/5, w + kk%5);
            else
                rc_X[i] = 0.0;
        }

        // register cache, no sync needed

        // Compute & Accumulate result of each tile (with warp-level shared)
        // int tid, val;
        #pragma unroll
        for (j = 0; j < 3; j++) {
            #pragma unroll
            for (i = 0; i < 16; i++) {
                // tid = i % 2; // thread ty in the warp to be queried
                // val = i / 2; // register value id of the query thread
                // rc_result[j] += __shfl_sync(mask, rc_W[j], ty * 16 + i) * __shfl_sync(mask, rc_X[val], tid * 16 + tx);
                rc_result[j] += __shfl_sync(mask, rc_W[j], ty * 16 + i) * __shfl_sync(mask, rc_X[i/2], i%2 * 16 + tx);
            }
        }

    }

    // Write
    #pragma unroll
    for (j = 0; j < 3; j++) {
        Row = Row_base + j * blockDim.y;
        Col = Col_base;
        if (Row < 6 && Col < 1936)
            y[b * 11616 + Row * 1936 + Col] = rc_result[j];
    }

#undef x4d
}

// Version 2 (2.3ms): 2x16 block, 6x32 output tile, 6x16 W tile, 16x32 X tile
__global__ void layer_1_6x32(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {

#define x4d(i3, i2, i1, i0) x[(i3) * 2304 + (i2) * 2304 + (i1) * 48 + i0]

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row_base = blockIdx.y * (3 * blockDim.y) + threadIdx.y;
    int Col_base = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    // rc for register cache
    float rc_W[3];
    float rc_X[8*2];
    float rc_result[3*2] = { 0 }; // zero out

    int m, Row, Col, Row_x, c, kk, h, w, i, j, t, n;
    unsigned mask = 0xffffffff; // __activemask();

    #pragma unroll
    for (t = 0; t < 2; t++) {
        m = t * 16; // Col in W & Row in X

        // Load tileW & tileX into register cache
        // W
        #pragma unroll
        for (i = 0; i < 3; i++) {
            Row = Row_base + i * blockDim.y;
            if (Row < 6 && m + tx < 25)
                rc_W[i] = k[Row*25 + (m + tx)];
            else
                rc_W[i] = 0.0;
        }
        // X
        #pragma unroll
        for (i = 0; i < 8; i++) {
            Row_x = m + i * blockDim.y + ty;
            c = Row_x / 25;
            kk = Row_x % 25;
            for (j = 0; j < 2; j++) {
                Col = Col_base + j * blockDim.x;
                h = Col / 44;
                w = Col % 44;
                if (Row_x < 25 && Col < 1936)
                    rc_X[i * 2 + j] = x4d(b, c, h + kk/5, w + kk%5);
                else
                    rc_X[i * 2 + j] = 0.0;
            }
        }

        // register cache, no sync needed

        // Compute & Accumulate result of each tile (with warp-level shared)
        // int tid, val;
        #pragma unroll
        for (i = 0; i < 3; i++) {
            #pragma unroll
            for (j = 0; j < 2; j++) {
                #pragma unroll
                for (n = 0; n < 16; n++) {
                    // tid = n % 2; // thread ty in the warp to be queried
                    // val = (n / 2) * 2 + j; // register value id of the query thread
                    rc_result[i * 2 + j] += __shfl_sync(mask, rc_W[i], ty * 16 + n) * __shfl_sync(mask, rc_X[n/2*2+j], n%2 * 16 + tx);
                }
            }
        }

    }

    // Write
    #pragma unroll
    for (i = 0; i < 3; i++) {
        Row = Row_base + i * blockDim.y;
        #pragma unroll
        for (j = 0; j < 2; j++) {
            Col = Col_base + j * blockDim.x;
            if (Row < 6 && Col < 1936)
                y[b * 11616 + Row * 1936 + Col] = rc_result[i * 2 + j];
        }
    }

#undef x4d
}

// Version 2.5: similar to Version 1 & 2, but reload N times and reuse W
#define N1 2
__global__ void layer_1_6x16N_reload(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {

#define x4d(i3, i2, i1, i0) x[(i3) * 2304 + (i2) * 2304 + (i1) * 48 + i0]

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row_base = blockIdx.y * (3 * blockDim.y) + threadIdx.y;
    int Col_base = blockIdx.x * (N1 * blockDim.x) + threadIdx.x;
    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    // rc for register cache
    float rc_W[3];
    float rc_X[8];
    float rc_result[3*N1] = { 0 }; // zero out

    int m, Row, Col, Row_x, c, kk, h, w, i, j, t, n;
    unsigned mask = 0xffffffff; // __activemask();

    #pragma unroll
    for (t = 0; t < 2; t++) {
        m = t * 16; // Col in W & Row in X

        // Load tileW & tileX into register cache
        // W
        #pragma unroll
        for (i = 0; i < 3; i++) {
            Row = Row_base + i * blockDim.y;
            if (Row < 6 && m + tx < 25)
                rc_W[i] = k[Row*25 + (m + tx)];
            else
                rc_W[i] = 0.0;
        }
        // X
        #pragma unroll
        for (j = 0; j < N1; j++) {
            #pragma unroll
            for (i = 0; i < 8; i++) {
                Row_x = m + i * blockDim.y + ty;
                c = Row_x / 25;
                kk = Row_x % 25;

                Col = Col_base + j * blockDim.x;
                h = Col / 44;
                w = Col % 44;
                if (Row_x < 25 && Col < 1936)
                    rc_X[i] = x4d(b, c, h + kk/5, w + kk%5);
                else
                    rc_X[i] = 0.0;
            }

            // register cache, no sync needed

            // Compute & Accumulate result of each tile (with warp-level shared)
            // int tid, val;
            #pragma unroll
            for (i = 0; i < 3; i++) {
                #pragma unroll
                for (n = 0; n < 16; n++) {
                    // tid = n % 2; // thread ty in the warp to be queried
                    // val = n / 2; // register value id of the query thread
                    rc_result[i * N1 + j] += __shfl_sync(mask, rc_W[i], ty * 16 + n) * __shfl_sync(mask, rc_X[n/2], n%2 * 16 + tx);
                }
            }
        }

    }

    // Write
    #pragma unroll
    for (i = 0; i < 3; i++) {
        Row = Row_base + i * blockDim.y;
        #pragma unroll
        for (j = 0; j < N1; j++) {
            Col = Col_base + j * blockDim.x;
            if (Row < 6 && Col < 1936)
                y[b * 11616 + Row * 1936 + Col] = rc_result[i * N1 + j];
        }
    }

#undef x4d
}

__global__ void layer_1_6x16N_cache(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {

#define x4d(i3, i2, i1, i0) x[(i3) * 2304 + (i2) * 2304 + (i1) * 48 + i0]

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row_base = blockIdx.y * (3 * blockDim.y) + threadIdx.y;
    int Col_base = blockIdx.x * (N1 * blockDim.x) + threadIdx.x;
    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    // rc for register cache
    float rc_W[3];
    float rc_X[8*N1];
    float rc_result[3*N1] = { 0 }; // zero out

    int m, Row, Col, Row_x, c, kk, h, w, i, j, t, n;
    unsigned mask = 0xffffffff; // __activemask();

    #pragma unroll
    for (t = 0; t < 2; t++) {
        m = t * 16; // Col in W & Row in X

        // Load tileW & tileX into register cache
        // W
        #pragma unroll
        for (i = 0; i < 3; i++) {
            Row = Row_base + i * blockDim.y;
            if (Row < 6 && m + tx < 25)
                rc_W[i] = k[Row*25 + (m + tx)];
            else
                rc_W[i] = 0.0;
        }
        // X
        #pragma unroll
        for (i = 0; i < 8; i++) {
            Row_x = m + i * blockDim.y + ty;
            c = Row_x / 25;
            kk = Row_x % 25;
            #pragma unroll
            for (j = 0; j < N1; j++) {
                Col = Col_base + j * blockDim.x;
                h = Col / 44;
                w = Col % 44;
                if (Row_x < 25 && Col < 1936)
                    rc_X[i * N1 + j] = x4d(b, c, h + kk/5, w + kk%5);
                else
                    rc_X[i * N1 + j] = 0.0;
            }
        }

        // register cache, no sync needed

        // Compute & Accumulate result of each tile (with warp-level shared)
        // int tid, val;
        #pragma unroll
        for (i = 0; i < 3; i++) {
            #pragma unroll
            for (j = 0; j < N1; j++) {
                #pragma unroll
                for (n = 0; n < 16; n++) {
                    // tid = n % 2; // thread ty in the warp to be queried
                    // val = (n / 2) * 2 + j; // register value id of the query thread
                    rc_result[i * N1 + j] += __shfl_sync(mask, rc_W[i], ty * 16 + n) * __shfl_sync(mask, rc_X[n/2*N1+j], n%2 * 16 + tx); // the "2" in n/2 is block width_y
                }
            }
        }

    }

    // Write
    #pragma unroll
    for (i = 0; i < 3; i++) {
        Row = Row_base + i * blockDim.y;
        #pragma unroll
        for (j = 0; j < N1; j++) {
            Col = Col_base + j * blockDim.x;
            if (Row < 6 && Col < 1936)
                y[b * 11616 + Row * 1936 + Col] = rc_result[i * N1 + j];
        }
    }

#undef x4d
}

// ------------------------------- Layer 2 ---------------------------------- //
// Version 0 (4.1 ms): 4x8 block, 16x16 output tile, 16x16 W tile, 16x16 X tile.
#define TILE_WIDTH_Y_2 (4)
#define TILE_WIDTH_X_2 (8)
#define TILE_LOAD_Y_2 (4)
#define TILE_LOAD_X_2 (2)
__global__ void layer_2_16x16(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {

#define x4d(i3, i2, i1, i0) x[(i3) * 2904 + (i2) * 484 + (i1) * 22 + i0]

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row_base = blockIdx.y * (TILE_LOAD_Y_2 * blockDim.y) + threadIdx.y;
    int Col_base = blockIdx.x * (TILE_LOAD_X_2 * blockDim.x) + threadIdx.x;
    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    // rc for register cache
    float rc_W[TILE_LOAD_Y_2 * TILE_LOAD_X_2];
    float rc_X[TILE_LOAD_Y_2 * TILE_LOAD_X_2];
    float rc_result[TILE_LOAD_Y_2 * TILE_LOAD_X_2] = { 0 }; // zero out

    int m, Row, Col, Row_x, c, kk, h, w, id, id_y, id_x, i, j, t, n;
    unsigned mask = 0xffffffff; // __activemask();

    #pragma unroll
    for (t = 0; t < 10; t++) {
        m = t * TILE_LOAD_X_2 * TILE_WIDTH_X_2; // Col in W & Row in X

        // Load tileW & tileX into register cache
        #pragma unroll
        for (i = 0; i < TILE_LOAD_Y_2; i++) {
            Row = Row_base + i * TILE_WIDTH_Y_2; // coarsened row index
            Row_x = m + i * TILE_WIDTH_Y_2 + ty; // same for each j iteration
            c = Row_x / 25;
            kk = Row_x % 25; // recover linearized KxK mask index
            #pragma unroll
            for (j = 0; j < TILE_LOAD_X_2; j++) {
                id = i * TILE_LOAD_X_2 + j;
                // Load W
                if (/*Row < 16 && */m + j * TILE_WIDTH_X_2 + tx < 150)
                    rc_W[id] = k[Row*150 + (m + j * TILE_WIDTH_X_2 + tx)];
                else
                    rc_W[id] = 0.0;

                // Load X
                Col = Col_base + j * TILE_WIDTH_X_2;
                h = Col / 18; // recover pixel location (h, w) in the input images
                w = Col % 18;
                if (Row_x < 150 && Col < 324)
                    rc_X[id] = x4d(b, c, h + kk/5, w + kk%5);
                else
                    rc_X[id] = 0.0;
            }
        }

        // register cache, no sync needed

        // Compute & Accumulate result of each tile (with warp-level shared)
        #pragma unroll
        for (i = 0; i < TILE_LOAD_Y_2; i++) { // when i = 0, access rc_W[0/1]; i = 1, access rc_W[2/3]; i = 2, access rc_W[4/5]; i = 3, access rc_W[6/7];
            #pragma unroll
            for (j = 0; j < TILE_LOAD_X_2; j++) { // when j = 0, access rc_X[0/2/4/6]; j = 1, access rc_X[1/3/5/7]
                id = i * TILE_LOAD_X_2 + j;
                #pragma unroll
                for (n = 0; n < TILE_WIDTH_Y_2; n++) {
                    id_y = ty * TILE_WIDTH_X_2 + (ty + n) % TILE_WIDTH_Y_2;
                    id_x = (ty + n) % TILE_WIDTH_Y_2 * TILE_WIDTH_X_2 + tx;
                    rc_result[id] += __shfl_sync(mask, rc_W[2*i], id_y)                * __shfl_sync(mask, rc_X[j], id_x);
                    rc_result[id] += __shfl_sync(mask, rc_W[2*i], id_y + TILE_WIDTH_Y_2) * __shfl_sync(mask, rc_X[j+2], id_x);
                    rc_result[id] += __shfl_sync(mask, rc_W[2*i+1], id_y)                * __shfl_sync(mask, rc_X[j+4], id_x);
                    rc_result[id] += __shfl_sync(mask, rc_W[2*i+1], id_y + TILE_WIDTH_Y_2) * __shfl_sync(mask, rc_X[j+6], id_x);
                }
            }
        }

    }

    // Write
    #pragma unroll
    for (i = 0; i < TILE_LOAD_Y_2; i++) {
        Row = Row_base + i * TILE_WIDTH_Y_2;
        #pragma unroll
        for (j = 0; j < TILE_LOAD_X_2; j++) {
            Col = Col_base + j * TILE_WIDTH_X_2;
            if (/*Row < 16 && */Col < 324)
                y[b * 5184 + Row * 324 + Col] = rc_result[i * TILE_LOAD_X_2 + j];
        }
    }

#undef x4d
}

// Version 1 (4.0 ms): 2x16 block, 16x16 output tile, 16x16 W tile, 16x16 X tile.
#define N2 2
__global__ void layer_2_16x16N_cache(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {

#define x4d(i3, i2, i1, i0) x[(i3) * 2904 + (i2) * 484 + (i1) * 22 + i0]

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row_base = blockIdx.y * (8 * blockDim.y) + threadIdx.y;
    int Col_base = blockIdx.x * (N2 * blockDim.x) + threadIdx.x;
    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    // rc for register cache
    float rc_W[8];
    float rc_X[8*N2];
    float rc_result[8*N2] = { 0 }; // zero out

    int m, Row, Col, Row_x, c, kk, h, w, i, j, t, n;
    unsigned mask = 0xffffffff; // __activemask();

    #pragma unroll
    for (t = 0; t < 10; t++) {
        m = t * 16; // Col in W & Row in X

        // Load tileW & tileX into register cache
        // W
        #pragma unroll
        for (i = 0; i < 8; i++) {
            Row = Row_base + i * blockDim.y;
            if (Row < 16 && m + tx < 150)
                rc_W[i] = k[Row*150 + (m + tx)];
            else
                rc_W[i] = 0.0;
        }
        // X
        #pragma unroll
        for (i = 0; i < 8; i++) {
            Row_x = m + i * blockDim.y + ty;
            c = Row_x / 25;
            kk = Row_x % 25;
            #pragma unroll
            for (j = 0; j < N2; j++) {
                Col = Col_base + j * blockDim.x;
                h = Col / 18;
                w = Col % 18;
                if (Row_x < 150 && Col < 324)
                    rc_X[i * N2 + j] = x4d(b, c, h + kk/5, w + kk%5);
                else
                    rc_X[i * N2 + j] = 0.0;
            }
        }

        // register cache, no sync needed

        // Compute & Accumulate result of each tile (with warp-level shared)
        // int tid, val;
        #pragma unroll
        for (i = 0; i < 8; i++) {
            #pragma unroll
            for (j = 0; j < N2; j++) {
                #pragma unroll
                for (n = 0; n < 16; n++) {
                    // tid = n % 2; // thread ty in the warp to be queried
                    // val = (n / 2) * 2 + j; // register value id of the query thread
                    rc_result[i * N2 + j] += __shfl_sync(mask, rc_W[i], ty * 16 + n) * __shfl_sync(mask, rc_X[n/2*N2+j], n%2 * 16 + tx); // the "2" in n/2 is block width_y
                }
            }
        }

    }

    // Write
    #pragma unroll
    for (i = 0; i < 8; i++) {
        Row = Row_base + i * blockDim.y;
        #pragma unroll
        for (j = 0; j < N2; j++) {
            Col = Col_base + j * blockDim.x;
            if (Row < 16 && Col < 324)
                y[b * 5184 + Row * 324 + Col] = rc_result[i * N2 + j];
        }
    }

#undef x4d
}

#elif OPTIMIZATION == 60

// ------------------------------- Layer 1 ---------------------------------- //
// Register cache for both W and X & half precision (2.1 ms)
__global__ void layer_1_register_half(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {

#define x4d(i3, i2, i1, i0) x[(i3) * 2304 + (i2) * 2304 + (i1) * 48 + i0]

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row_base = blockIdx.y * (3 * blockDim.y) + threadIdx.y;
    int Col_base = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    // rc for register cache
    half2 rc_W[3];
    half2 rc_X[8];
    half2 rc_result[3] = { __float2half2_rn(0) }; // zero out

    int m, Row, Col, Row_x, c, kk, i, t, n;
    unsigned mask = 0xffffffff; // __activemask();
    half temp1, temp2;

    #pragma unroll
    for (t = 0; t < 2; t++) {
        m = t * 16; // Col in W & Row in X

        // Load tileW & tileX into register cache
        // W
        #pragma unroll
        for (i = 0; i < 3; i++) {
            Row = Row_base + i * blockDim.y;
            if (Row < 6 && m + tx < 25)
                rc_W[i] = __half2half2(__float2half(k[Row*25 + (m + tx)]));
            else
                rc_W[i] = __float2half2_rn(0);
        }
        // X
        #pragma unroll
        for (i = 0; i < 8; i++) {
            Row_x = m + i * blockDim.y + ty;
            c = Row_x / 25;
            kk = Row_x % 25;

            Col = Col_base;
            if (Row_x < 25 && Col < 1936)
                temp1 = __float2half(x4d(b, c, Col/44 + kk/5, Col%44 + kk%5));
            else
                temp1 = __float2half(0);

            Col = Col_base + blockDim.x;
            if (Row_x < 25 && Col < 1936)
                temp2 = __float2half(x4d(b, c, Col/44 + kk/5, Col%44 + kk%5));
            else
                temp2 = __float2half(0);

            rc_X[i] = __halves2half2(temp1, temp2);
        }

        // register cache, no sync needed

        // Compute & Accumulate result of each tile (with warp-level shared)
        // int tid, val;
        #pragma unroll
        for (i = 0; i < 3; i++) {
            #pragma unroll
            for (n = 0; n < 16; n++) {
                // tid = n % 2; // thread ty in the warp to be queried
                // val = (n / 2) * 2 + j; // register value id of the query thread
                rc_result[i] = __hfma2(__shfl_sync(mask, rc_W[i], ty * 16 + n), __shfl_sync(mask, rc_X[n/2], n%2 * 16 + tx), rc_result[i]);
            }
        }

    }

    // Write
    #pragma unroll
    for (i = 0; i < 3; i++) {
        Row = Row_base + i * blockDim.y;
        if (Row < 6 && Col_base < 1936)
            y[b * 11616 + Row * 1936 + Col_base] = __low2float(rc_result[i]);
        if (Row < 6 && Col_base + blockDim.x < 1936)
            y[b * 11616 + Row * 1936 + Col_base + blockDim.x] = __high2float(rc_result[i]);
    }

#undef x4d
}

// Shared memory for both W and X & thread coarsening & half precision (1.9ms)
__global__ void layer_1_shmem_half(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {

#define x4d(i3, i2, i1, i0) x[(i3) * 2304 + (i2) * 2304 + (i1) * 48 + i0]

    __shared__ half2 W_shared[6][25];
    __shared__ half2 X_shared[25][88];

    int tz = threadIdx.z;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int b = blockIdx.z;
    int Col_base = blockIdx.x * (2 * blockDim.x * blockDim.z) + tz * (2 * blockDim.x) + tx;

    int Row_x, i, n, idx;

    // Load W (only the first 6x22 threads are needed)
    if (tz == 0) {
        W_shared[ty][tx] = __half2half2(__float2half(k[ty*25 + tx]));
        // Extra load W
        if (tx < 3)
            W_shared[ty][tx+22] = __half2half2(__float2half(k[ty*25 + tx + 22]));
    }

    // Load X
    int h = 4 * blockIdx.x + tz;
    #pragma unroll
    for (i = 0; i < 4; i++) {
        Row_x = i * 6 + ty;
        // Col = Col_base + tz * 44;
        idx = b * 2304 + (h + Row_x/5) * 48 + tx + Row_x%5;
        // X_shared[Row_x][tz*22+tx] = __floats2half2_rn(x4d(b, 0, h + Row_x/5, tx + Row_x%5), x4d(b, 0, h + Row_x/5, tx + 22 + Row_x%5));
        X_shared[Row_x][tz*22+tx] = __floats2half2_rn(x[idx], x[idx+22]);
    }
    // Extra load X
    if (ty == 0) {
        idx = b * 2304 + (h + 4) * 48 + tx + 4;
        // X_shared[24][tz*22+tx] = __floats2half2_rn(x4d(b, 0, h + 4, tx + 4), x4d(b, 0, h + 4, tx + 22 + 4));
        X_shared[24][tz*22+tx] = __floats2half2_rn(x[idx], x[idx+22]);
    }

    __syncthreads();

    // Compute
    half2 result = __float2half2_rn(0);
    #pragma unroll
    for (n = 0; n < 25; n++)
        result = __hfma2(/*__half2half2(__float2half(k[ty*25 + n]))*/W_shared[ty][n], X_shared[n][tz*22+tx], result);

    // Write
    idx = b * 11616 + ty * 1936 + Col_base;
    y[idx] = __low2float(result);
    y[idx + 22] = __high2float(result);

#undef x4d
}

// ------------------------------- Layer 2 ---------------------------------- //
// Version 2 (3.8 ms): half precision
__global__ void layer_2_16x32_half(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {

#define x4d(i3, i2, i1, i0) x[(i3) * 2904 + (i2) * 484 + (i1) * 22 + i0]

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row_base = blockIdx.y * (8 * blockDim.y) + threadIdx.y;
    int Col_base = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    // rc for register cache
    half2 rc_W[8];
    half2 rc_X[8];
    half2 rc_result[8] = { __float2half2_rn(0) }; // zero out

    int m, Row, Col, Row_x, c, kk, i, t, n;
    unsigned mask = 0xffffffff; // __activemask();
    half temp1, temp2;

    #pragma unroll
    for (t = 0; t < 10; t++) {
        m = t * 16; // Col in W & Row in X

        // Load tileW & tileX into register cache
        // W
        #pragma unroll
        for (i = 0; i < 8; i++) {
            Row = Row_base + i * blockDim.y;
            if (Row < 16 && m + tx < 150)
                rc_W[i] = __half2half2(__float2half(k[Row*150 + (m + tx)]));
            else
                rc_W[i] = __float2half2_rn(0);
        }
        // X
        #pragma unroll
        for (i = 0; i < 8; i++) {
            Row_x = m + i * blockDim.y + ty;
            c = Row_x / 25;
            kk = Row_x % 25;

            Col = Col_base;
            if (Row_x < 150 && Col < 324)
                temp1 = __float2half(x4d(b, c, Col/18 + kk/5, Col%18 + kk%5));
            else
                temp1 = __float2half(0);

            Col = Col_base + blockDim.x;
            if (Row_x < 150 && Col < 324)
                temp2 = __float2half(x4d(b, c, Col/18 + kk/5, Col%18 + kk%5));
            else
                temp2 = __float2half(0);

            rc_X[i] = __halves2half2(temp1, temp2);
        }

        // register cache, no sync needed

        // Compute & Accumulate result of each tile (with warp-level shared)
        // int tid, val;
        #pragma unroll
        for (i = 0; i < 8; i++) {
            #pragma unroll
            for (n = 0; n < 16; n++) {
                // tid = n % 2; // thread ty in the warp to be queried
                // val = (n / 2) * 2 + j; // register value id of the query thread
                rc_result[i] = __hfma2(__shfl_sync(mask, rc_W[i], ty * 16 + n), __shfl_sync(mask, rc_X[n/2], n%2 * 16 + tx), rc_result[i]);
            }
        }

    }

    // Write
    #pragma unroll
    for (i = 0; i < 8; i++) {
        Row = Row_base + i * blockDim.y;
        if (Row < 16 && Col_base < 324)
            y[b * 5184 + Row * 324 + Col_base] = __low2float(rc_result[i]);
        if (Row < 16 && Col_base + blockDim.x < 324)
            y[b * 5184 + Row * 324 + Col_base + blockDim.x] = __high2float(rc_result[i]);
    }

#undef x4d
}
#elif OPTIMIZATION == 61

#define TILE_WIDTH (22)
#define BLOCK_WIDTH (TILE_WIDTH + 4)
__global__ void layer_1(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k){

#define y4d(i3, i2, i1, i0) y[(i3) * 11616 + (i2) * 1936 + (i1) * 44 + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * 2304 + (i2) * 2304 + (i1) * 48 + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * 25 + (i2) * 25 + (i1) * 5 + i0]

    __shared__ float x_shared[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ float k_shared[5][5];

    int b = blockIdx.x; // batch
    int m = blockIdx.y; //output feature maps
    int c = 0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int W_grid = ceil((float)44 / TILE_WIDTH);
    // Output index
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;

    // Load X
    x_shared[ty][tx] = h < 48 && w < 48 ? x4d(b, c, h, w) : 0;

    // Load W
    if (ty < 5 && tx < 5)
        k_shared[ty][tx] = k4d(m, c, ty, tx);

    __syncthreads();

    // Strategy 2: all load, part compute & write
    if (ty < 22 && tx < 22 && h < 44 && w < 44) {
        // Compute
        float result = 0;
        for (int p = 0; p < 5; ++p)
      	    for (int q = 0; q < 5; ++q)
      	        result += x_shared[ty+p][tx+q] * k_shared[p][q];

        // Write
        y4d(b, m, h, w) = result;
    }

#undef y4d
#undef x4d
#undef k4d
}

__global__ void layer_1_coarsen(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k){

#define y4d(i3, i2, i1, i0) y[(i3) * 11616 + (i2) * 1936 + (i1) * 44 + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * 2304 + (i2) * 2304 + (i1) * 48 + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * 25 + (i2) * 25 + (i1) * 5 + i0]

    __shared__ float x_shared[48][48];
    float k_shared[25];

    int b = blockIdx.x; // batch
    int m = blockIdx.y; //output feature maps
    int c = 0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int h, w, i, j, p, q;
    // Full Load X
    #pragma unroll
    for (i = 0; i < 2; i++) {
        h = i * BLOCK_WIDTH + ty;
        #pragma unroll
        for (j = 0; j < 2; j++) {
            w = j * BLOCK_WIDTH + tx;
            if (h < 48 && w < 48) {
                x_shared[h][w] = x4d(b, c, h, w);
            }
        }
    }

    // Load X
    #pragma unroll
    for (p = 0; p < 5; p++) {
        #pragma unroll
        for (q = 0; q < 5; q++) {
            k_shared[p * 5 + q] = k4d(m, c, p, q);
        }
    }

    __syncthreads();

    // Compute
    float result;
    #pragma unroll
    for (i = 0; i < 2; i++) {
        h = i * TILE_WIDTH + ty;
        #pragma unroll
        for (j = 0; j < 2; j++) {
            w = j * TILE_WIDTH + tx;
            if (h < 44 && w < 44) {
                // Compute
                result = 0;
                #pragma unroll
                for (p = 0; p < 5; ++p)
                    #pragma unroll
                    for (q = 0; q < 5; ++q)
                        result += x_shared[h+p][w+q] * k_shared[p * 5 + q];
                // Write
                y4d(b, m, h, w) = result;
            }
        }
    }

#undef y4d
#undef x4d
#undef k4d
}

__constant__ float k1[600];
__global__ void layer_1_half(float * __restrict__ y, float * __restrict__ x/*, float * __restrict__ k*/){

#define y4d(i3, i2, i1, i0) y[(i3) * 11616 + (i2) * 1936 + (i1) * 44 + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * 2304 + (i2) * 2304 + (i1) * 48 + i0]
#define k4d(i3, i2, i1, i0) k1[(i3) * 25 + (i2) * 25 + (i1) * 5 + i0]

    __shared__ half x_shared[48][48];
    half2 k_shared[25];

    int b = blockIdx.x; // batch
    int m = blockIdx.y; //output feature maps
    int c = 0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int h, i, p, q;

    // Full Load X
    #pragma unroll
    for (i = 0; i < 2; i++) {
        h = i * 24 + ty;
        x_shared[h][tx] = __float2half(x4d(b, c, h, tx));
        x_shared[h][tx+24] = __float2half(x4d(b, c, h, tx+24));
    }

    // Load W
    #pragma unroll
    for (p = 0; p < 5; p++) {
        #pragma unroll
        for (q = 0; q < 5; q++) {
            k_shared[p * 5 + q] = __half2half2(__float2half(k4d(m, c, p, q)));
        }
    }

    __syncthreads();

    half2 result;
    if (ty < 22 && tx < 22) {
        #pragma unroll
        for (i = 0; i < 2; i++) {
                result = __float2half2_rn(0);
                // Compute
                h = i * 22 + ty;
                #pragma unroll
                for (p = 0; p < 5; ++p)
                    #pragma unroll
                    for (q = 0; q < 5; ++q)
                        result = __hfma2(__halves2half2(x_shared[h+p][tx+q], x_shared[h+p][tx+22+q]), k_shared[p * 5 + q], result);
                // Write
                y4d(b, m, h, tx) = __low2float(result);
                y4d(b, m, h, tx+22) = __high2float(result);
        }
    }

#undef y4d
#undef x4d
#undef k4d
}

__constant__ float k2[2400];
__global__ void layer_2_half(float * __restrict__ y, float * __restrict__ x/*, float * __restrict__ k*/){

#define y4d(i3, i2, i1, i0) y[(i3) * 5184 + (i2) * 324 + (i1) * 18 + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * 2904 + (i2) * 484 + (i1) * 22 + i0]
#define k4d(i3, i2, i1, i0) k2[(i3) * 150 + (i2) * 25 + (i1) * 5 + i0]

    __shared__ half x_shared[22][22];
    half2 k_shared[25];

    int b = blockIdx.x; // batch
    int m = blockIdx.y; //output feature maps

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int h, i, p, q, c;

    half2 result[2];
    result[0] = __float2half2_rn(0);
    result[1] = __float2half2_rn(0);

    #pragma unroll
    for (c = 0; c < 6; c++) {
        // Full Load X
        #pragma unroll
        for (i = 0; i < 2; i++) {
            h = i * 11 + ty;
            x_shared[h][tx] = __float2half(x4d(b, c, h, tx));
            x_shared[h][tx+11] = __float2half(x4d(b, c, h, tx+11));
        }

        // Load W
        #pragma unroll
        for (p = 0; p < 5; p++) {
            #pragma unroll
            for (q = 0; q < 5; q++) {
                k_shared[p * 5 + q] = __half2half2(__float2half(k4d(m, c, p, q)));
            }
        }

        __syncthreads();

        // Compute
        if (ty < 9 && tx < 9) {
            #pragma unroll
            for (i = 0; i < 2; i++) {
                    h = i * 9 + ty;
                    #pragma unroll
                    for (p = 0; p < 5; ++p)
                        #pragma unroll
                        for (q = 0; q < 5; ++q)
                            result[i] = __hfma2(__halves2half2(x_shared[h+p][tx+q], x_shared[h+p][tx+9+q]), k_shared[p * 5 + q], result[i]);
            }
        }

        __syncthreads();
    }

    // Write
    if (ty < 9 && tx < 9) {
        y4d(b, m, ty, tx) = __low2float(result[0]);
        y4d(b, m, ty, tx+9) = __high2float(result[0]);
        y4d(b, m, ty+9, tx) = __low2float(result[1]);
        y4d(b, m, ty+9, tx+9) = __high2float(result[1]);
    }

#undef y4d
#undef x4d
#undef k4d
}

__global__ void layer_2_double_buffer(float * __restrict__ y, float * __restrict__ x/*, float * __restrict__ k*/){

#define y4d(i3, i2, i1, i0) y[(i3) * 5184 + (i2) * 324 + (i1) * 18 + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * 2904 + (i2) * 484 + (i1) * 22 + i0]
#define k4d(i3, i2, i1, i0) k2[(i3) * 150 + (i2) * 25 + (i1) * 5 + i0]

    __shared__ half x_shared_0[22][22];
    __shared__ half x_shared_1[22][22]; // double buffer
    half2 k_shared[25];

    half (*x_src)[22] = x_shared_0;
    half (*x_dst)[22] = x_shared_1;
    half (*temp)[22];

    int b = blockIdx.x; // batch
    int m = blockIdx.y; //output feature maps

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int h, i, p, q, c;

    half2 result[2];
    result[0] = __float2half2_rn(0);
    result[1] = __float2half2_rn(0);

    // Pre-load X for 1st run
    c = 0;
    #pragma unroll
    for (i = 0; i < 2; i++) {
        h = i * 11 + ty;
        x_src[h][tx] = __float2half(x4d(b, c, h, tx));
        x_src[h][tx+11] = __float2half(x4d(b, c, h, tx+11));
    }

    #pragma unroll
    for (c = 0; c < 6; c++) {

        __syncthreads();

        // Load W
        #pragma unroll
        for (p = 0; p < 5; p++) {
            #pragma unroll
            for (q = 0; q < 5; q++) {
                k_shared[p * 5 + q] = __half2half2(__float2half(k4d(m, c, p, q)));
            }
        }

        // Compute
        if (ty < 9 && tx < 9) {
            #pragma unroll
            for (i = 0; i < 2; i++) {
                    h = i * 9 + ty;
                    #pragma unroll
                    for (p = 0; p < 5; ++p)
                        #pragma unroll
                        for (q = 0; q < 5; ++q)
                            result[i] = __hfma2(__halves2half2(x_src[h+p][tx+q], x_src[h+p][tx+9+q]), k_shared[p * 5 + q], result[i]);
            }
        }

        if (c != 5) {
            // Pre-fetch (note the "c + 1"!)
            #pragma unroll
            for (i = 0; i < 2; i++) {
                h = i * 11 + ty;
                x_dst[h][tx] = __float2half(x4d(b, c+1, h, tx));
                x_dst[h][tx+11] = __float2half(x4d(b, c+1, h, tx+11));
            }

            // Swap
            temp = x_dst;
            x_dst = x_src;
            x_src = temp;
        }

    }

    // Write
    if (ty < 9 && tx < 9) {
        y4d(b, m, ty, tx) = __low2float(result[0]);
        y4d(b, m, ty, tx+9) = __high2float(result[0]);
        y4d(b, m, ty+9, tx) = __low2float(result[1]);
        y4d(b, m, ty+9, tx+9) = __high2float(result[1]);
    }

#undef y4d
#undef x4d
#undef k4d
}

#define N (1)
__global__ void layer_2_coarsen(float * __restrict__ y, float * __restrict__ x/*, float * __restrict__ k*/){

#define y4d(i3, i2, i1, i0) y[(i3) * 5184 + (i2) * 324 + (i1) * 18 + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * 2904 + (i2) * 484 + (i1) * 22 + i0]
#define k4d(i3, i2, i1, i0) k2[(i3) * 150 + (i2) * 25 + (i1) * 5 + i0]

    __shared__ half x_shared[22][22];
    half2 k_shared[25];

    int b_base = blockIdx.x * N; // batch
    int m = blockIdx.y; //output feature maps

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int h, i, p, q, c, b, batch;

    half2 result[N][2];
    for (i = 0; i < N; i++) {
        result[i][0] = __float2half2_rn(0);
        result[i][1] = __float2half2_rn(0);
    }

    #pragma unroll
    for (c = 0; c < 6; c++) {

        // Load W (use same W for all N batches)
        #pragma unroll
        for (p = 0; p < 5; p++) {
            #pragma unroll
            for (q = 0; q < 5; q++) {
                k_shared[p * 5 + q] = __half2half2(__float2half(k4d(m, c, p, q)));
            }
        }

        #pragma unroll
        for (batch = 0; batch < N; batch++) {
            b = b_base + batch;

            // Full Load X
            #pragma unroll
            for (i = 0; i < 2; i++) {
                h = i * 11 + ty;
                x_shared[h][tx] = __float2half(x4d(b, c, h, tx));
                x_shared[h][tx+11] = __float2half(x4d(b, c, h, tx+11));
            }

            __syncthreads();

            // Compute
            if (ty < 9 && tx < 9) {
                #pragma unroll
                for (i = 0; i < 2; i++) {
                        h = i * 9 + ty;
                        #pragma unroll
                        for (p = 0; p < 5; ++p)
                            #pragma unroll
                            for (q = 0; q < 5; ++q)
                                result[batch][i] = __hfma2(__halves2half2(x_shared[h+p][tx+q], x_shared[h+p][tx+9+q]), k_shared[p * 5 + q], result[batch][i]);
                }
            }

            __syncthreads();
        }

    }

    // Write
    if (ty < 9 && tx < 9) {
        #pragma unroll
        for (batch = 0; batch < N; batch++) {
            b = b_base + batch;
            y4d(b, m, ty, tx) = __low2float(result[batch][0]);
            y4d(b, m, ty, tx+9) = __high2float(result[batch][0]);
            y4d(b, m, ty+9, tx) = __low2float(result[batch][1]);
            y4d(b, m, ty+9, tx+9) = __high2float(result[batch][1]);
        }
    }

#undef y4d
#undef x4d
#undef k4d
}

#elif OPTIMIZATION == 6

// 2 Strategies:
//    a) local_shmem: use small shared memory for W, but reload in each iteration.
//    b) full_shmem: use large shared memory for W, preload all W to avoid __syncthreads

__global__ void layer_1_half_local_shmem(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {

#define x4d(i3, i2, i1, i0) x[(i3) * 2304 + (i2) * 2304 + (i1) * 48 + i0]

    __shared__ half2 W_shared[6][16];

    int tz = threadIdx.z;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row_base = blockIdx.y * (3 * blockDim.y) + threadIdx.y;
    int Col_base = blockIdx.x * (2 * blockDim.x * blockDim.z) + tz * (2 * blockDim.x) + threadIdx.x;
    // use z (batch No.) to offset indexing when loading X & writing Y
    int b = blockIdx.z;

    // rc for register cache
    half2 rc_X[8];
    half2 rc_result[3] = { __float2half2_rn(0) }; // zero out

    int Row, Col, Row_x, c, kk, i, t, m, n;
    unsigned mask = 0xffffffff; // __activemask();
    half temp1, temp2;

    #pragma unroll
    for (t = 0; t < 2; t++) {
        m = t * 16;

        // Load W into shared memory (only first eight 2x16 blocks will be loading)
        if (tz < 3) {
            Row = tz * blockDim.y + ty;
            if (/*Row < 6 && */m + tx < 25)
                W_shared[Row][tx] = __half2half2(__float2half(k[Row*25 + (m + tx)]));
            else
                W_shared[Row][tx] = __float2half2_rn(0);
        }

        __syncthreads();

        // Load X into register cache (each thread load two X elements and compress into half2)
        #pragma unroll
        for (i = 0; i < 8; i++) {
            Row_x = m + i * blockDim.y + ty;
            c = Row_x / 25;
            kk = Row_x % 25;

            Col = Col_base;
            if (Row_x < 25 && Col < 1936)
                temp1 = __float2half(x4d(b, c, Col/44 + kk/5, Col%44 + kk%5));
            else
                temp1 = __float2half(0);

            Col = Col_base + blockDim.x;
            if (Row_x < 25 && Col < 1936)
                temp2 = __float2half(x4d(b, c, Col/44 + kk/5, Col%44 + kk%5));
            else
                temp2 = __float2half(0);

            rc_X[i] = __halves2half2(temp1, temp2);
        }

        // register cache, no sync needed

        // Compute & Accumulate result of each tile (with warp-level shared)
        // int tid, val;
        #pragma unroll
        for (i = 0; i < 3; i++) {
            #pragma unroll
            for (n = 0; n < 16; n++) {
                // tid = n % 2; // thread ty in the warp to be queried
                // val = (n / 2) * 2 + j; // register value id of the query thread
                rc_result[i] = __hfma2(W_shared[i*2+ty][n], __shfl_sync(mask, rc_X[n/2], n%2 * 16 + tx), rc_result[i]);
            }
        }

        __syncthreads(); // necessary
    }

    // Write
    #pragma unroll
    for (i = 0; i < 3; i++) {
        Row = Row_base + i * blockDim.y;
        if (/*Row < 6 && */Col_base < 1936)
            y[b * 11616 + Row * 1936 + Col_base] = __low2float(rc_result[i]);
        if (/*Row < 6 && */Col_base + blockDim.x < 1936)
            y[b * 11616 + Row * 1936 + Col_base + blockDim.x] = __high2float(rc_result[i]);
    }

#undef x4d
}

__global__ void layer_1_half_full_shmem(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {

#define x4d(i3, i2, i1, i0) x[(i3) * 2304 + (i2) * 2304 + (i1) * 48 + i0]

    __shared__ half2 W_shared[6][32]; // pre-load everything

    int tz = threadIdx.z;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row_base = blockIdx.y * (3 * blockDim.y) + threadIdx.y;
    int Col_base = blockIdx.x * (2 * blockDim.x * blockDim.z) + tz * (2 * blockDim.x) + threadIdx.x;
    int b = blockIdx.z;

    // rc for register cache
    half2 rc_X[8];
    half2 rc_result[3] = { __float2half2_rn(0) }; // zero out

    int Row, Col, Row_x, kk, i, t, m, n, idx;
    half temp1, temp2;

    // Load W into shared memory (only first eight 2x16 blocks will be loading)
    if (tz < 2) {
        #pragma unroll
        for (i = 0; i < 3; ++i) {
            Row = i * blockDim.y + ty;
            Col = tz * blockDim.x + tx;

            if (/*Row < 6 && */Col < 25)
                W_shared[Row][Col] = __half2half2(__float2half(k[Row*25 + Col])); // same row in each ith iteration, coalescing!
            else
                W_shared[Row][Col] = __float2half2_rn(0);
        }
    }

    __syncthreads();

    #pragma unroll
    for (t = 0; t < 2; t++) {
        m = t * 16;

        // Load X into register cache (each thread load two X elements and compress into half2)
        #pragma unroll
        for (i = 0; i < 8; i++) {
            Row_x = m + i * blockDim.y + ty;
            kk = Row_x % 25;

            Col = Col_base;
            if (Row_x < 25 && Col < 1936)
                temp1 = __float2half(x[b*2304 + (Col/44 + kk/5)*48 + Col%44 + kk%5]);
            else
                temp1 = __float2half(0);

            Col = Col_base + blockDim.x;
            if (Row_x < 25 && Col < 1936)
                temp2 = __float2half(x[b*2304 + (Col/44 + kk/5)*48 + Col%44 + kk%5]);
            else
                temp2 = __float2half(0);

            rc_X[i] = __halves2half2(temp1, temp2);
        }

        // register cache, no sync needed

        // Compute & Accumulate result of each tile (with warp-level shared)
        // int tid, val;
        #pragma unroll
        for (i = 0; i < 3; i++) {
            #pragma unroll
            for (n = 0; n < 16; n++) {
                // tid = n % 2; // thread ty in the warp to be queried
                // val = (n / 2) * 2 + j; // register value id of the query thread
                rc_result[i] = __hfma2(W_shared[i*2+ty][m+n], __shfl_sync(0xffffffff, rc_X[n/2], n%2 * 16 + tx), rc_result[i]);
            }
        }

    }

    // Write
    #pragma unroll
    for (i = 0; i < 3; i++) {
        Row = Row_base + i * blockDim.y;
        idx = b * 11616 + Row * 1936 + Col_base;
        if (/*Row < 6 && */Col_base < 1936)
            y[idx] = __low2float(rc_result[i]);
        if (/*Row < 6 && */Col_base + blockDim.x < 1936)
            y[idx + blockDim.x] = __high2float(rc_result[i]);
    }

#undef x4d
}

__global__ void layer_2_half_local_shmem(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {

#define x4d(i3, i2, i1, i0) x[(i3) * 2904 + (i2) * 484 + (i1) * 22 + i0]

    __shared__ half2 W_shared[16][16]; // half2 collapses 16x32 region to 16x16

    int tz = threadIdx.z;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row_base = blockIdx.y * (8 * blockDim.y) + threadIdx.y; // 8x load in row direction
    int Col_base = blockIdx.x * (2 * blockDim.x * blockDim.z) + tz * (2 * blockDim.x) + threadIdx.x; // 2x load in column direction
    int b = blockIdx.z;

    // rc for register cache
    half2 rc_X[8];
    half2 rc_result[8] = { __float2half2_rn(0) }; // zero out

    int Row, Col, Row_x, c, kk, i, t, m, n, idx;
    half temp1, temp2;
    half zero = __float2half(0);

    #pragma unroll
    for (t = 0; t < 10; t++) {
        m = t * 16;

        // Load W into shared memory (only first eight 2x16 blocks will be loading)
        if (tz < 8) {
            Row = tz * blockDim.y + ty;
            if (/*Row < 16 && */m + tx < 150)
                W_shared[Row][tx] = __half2half2(__float2half(k[Row*150 + (m + tx)]));
            else
                W_shared[Row][tx] = __float2half2_rn(0);
        }

        // Important note: where you put sync() affects performance! if I move sync down after X load, it will be slowed down by 1ms...
        __syncthreads(); // W in shmem, need sync.

        // Load X into register cache (each thread load two X elements and compress into half2)
        #pragma unroll
        for (i = 0; i < 8; i++) {
            Row_x = m + i * blockDim.y + ty;
            c = Row_x / 25;
            kk = Row_x % 25;

            idx = b * 2904 + c * 484;
            temp1 = zero;
            temp2 = zero;
            // 1st element
            Col = Col_base;
            if (Row_x < 150 && Col < 324)
                temp1 = __float2half(x[idx + (Col/18 + kk/5) * 22 + Col%18 + kk%5]);
            // 2nd element
            Col = Col_base + blockDim.x;
            if (Row_x < 150 && Col < 324)
                temp2 = __float2half(x[idx + (Col/18 + kk/5) * 22 + Col%18 + kk%5]);

            rc_X[i] = __halves2half2(temp1, temp2);
        }

        // X in register cache, no sync needed

        // Compute & Accumulate result of each tile (with warp-level shared)
        #pragma unroll
        for (i = 0; i < 8; i++) {
            #pragma unroll
            for (n = 0; n < 16; n++) {
                // int tid, val;
                // tid = n % 2; // thread ty in the warp to be queried
                // val = (n / 2) * 2 + j; // register value id of the query thread
                rc_result[i] = __hfma2(W_shared[i*2+ty][n], __shfl_sync(0xffffffff, rc_X[n/2], n%2 * 16 + tx), rc_result[i]);
            }
        }

        __syncthreads(); // necessary
    }

    // Write
    #pragma unroll
    for (i = 0; i < 8; i++) {
        Row = Row_base + i * blockDim.y;
        idx = b * 5184 + Row * 324 + Col_base;
        if (/*Row < 16 && */Col_base < 324)
            y[idx] = __low2float(rc_result[i]);
        if (/*Row < 16 && */Col_base + blockDim.x < 324)
            y[idx + blockDim.x] = __high2float(rc_result[i]);
    }

#undef x4d
}

__global__ void layer_2_half_full_shmem(float * __restrict__ y, float * __restrict__ x, float * __restrict__ k) {

#define x4d(i3, i2, i1, i0) x[(i3) * 2904 + (i2) * 484 + (i1) * 22 + i0]

    __shared__ half2 W_shared[16][160]; // pre-load everything, only 10kb

    int tz = threadIdx.z;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int Row_base = blockIdx.y * (8 * blockDim.y) + threadIdx.y; // 8x load in row direction
    int Col_base = blockIdx.x * (2 * blockDim.x * blockDim.z) + tz * (2 * blockDim.x) + threadIdx.x; // 2x load in column direction
    int b = blockIdx.z;

    // rc for register cache
    half2 rc_X[8];
    half2 rc_result[8] = { __float2half2_rn(0) }; // zero out

    int Row, Col, Row_x, c, kk, i, t, m, n, idx;
    half temp1, temp2;
    half zero = __float2half(0);

    // Pre-load W into shared memory (be careful of memory coalescing!)
    if (tz < 10) { // only 10 out of 12 2x16 blocks participate
        #pragma unroll
        for (i = 0; i < 8; ++i) {
            Row = i * blockDim.y + ty;
            Col = tz * blockDim.x + tx;

            if (/*Row < 16 && */Col < 150)
                W_shared[Row][Col] = __half2half2(__float2half(k[Row*150 + Col])); // same row in each ith iteration, coalescing!
            else
                W_shared[Row][Col] = __float2half2_rn(0);
        }
    }

    __syncthreads(); // W in shmem, need sync.

    #pragma unroll
    for (t = 0; t < 10; ++t) {
        m = t * 16;

        // Load X into register cache (each thread load two X elements and compress into half2)
        #pragma unroll
        for (i = 0; i < 8; ++i) {
            Row_x = m + i * blockDim.y + ty;
            c = Row_x / 25;
            kk = Row_x % 25;

            idx = b * 2904 + c * 484;
            temp1 = zero;
            temp2 = zero;
            // 1st element
            Col = Col_base;
            if (Row_x < 150 && Col < 324)
                temp1 = __float2half(x[idx + (Col/18 + kk/5) * 22 + Col%18 + kk%5]);
            // 2nd element
            Col = Col_base + blockDim.x;
            if (Row_x < 150 && Col < 324)
                temp2 = __float2half(x[idx + (Col/18 + kk/5) * 22 + Col%18 + kk%5]);

            rc_X[i] = __halves2half2(temp1, temp2);
        }

        // X in register cache, no sync needed

        // Compute & Accumulate result of each tile (with warp-level shared)
        #pragma unroll
        for (i = 0; i < 8; ++i) {
            #pragma unroll
            for (n = 0; n < 16; ++n) {
                // int tid, val;
                // tid = n % 2; // thread ty in the warp to be queried
                // val = (n / 2) * 2 + j; // register value id of the query thread
                rc_result[i] = __hfma2(W_shared[i*2+ty][m+n], __shfl_sync(0xffffffff, rc_X[n/2], n%2 * 16 + tx), rc_result[i]);
            }
        }

    }

    // Write
    #pragma unroll
    for (i = 0; i < 8; ++i) {
        Row = Row_base + i * blockDim.y;
        idx = b * 5184 + Row * 324 + Col_base;
        if (/*Row < 16 && */Col_base < 324)
            y[idx] = __low2float(rc_result[i]);
        if (/*Row < 16 && */Col_base + blockDim.x < 324)
            y[idx + blockDim.x] = __high2float(rc_result[i]);
    }

#undef x4d
}
#endif

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

#if QUERY == 1
    // Device Query
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("====== Device Query\n");
    printf("%d device(s) supporting CUDA\n", deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        printf("=== Device %d\n", dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        deviceQuery(deviceProp);
    }
    printf("====== End Device Query\n");
#endif

    // Extract the tensor dimensions into B,M,C,H,W,K
    /*
    Data layout:
    y: output data, batch size * output channels * y * x, [B][M][H-K+1][W-K+1]
    x: input data, batch size * input channels * y * x, [B][C][H][W]
    k: kernel weights, output channels * input channels * y * x, [M][C][K][K]
    in this project, all 1st layer images are 48x48, all 2nd layer images are 22x22, all mask are 5x5, if hard-code needed
    */
#if OPTIMIZATION != 4 && OPTIMIZATION != 5 && OPTIMIZATION != 6 // 4 & 5 & 6 are hard-coded
    const int B = y.shape_[0]; // batch size
    const int M = y.shape_[1]; // output channels
    const int C = x.shape_[1]; // input channels
    const int H = x.shape_[2]; // image height
    const int W = x.shape_[3]; // image width
    const int K = w.shape_[2]; // kernel size
    // printf("B = %d, M = %d, C = %d, H = %d, W = %d, K = %d\n",B,M,C,H,W,K);
    // 1st layer: B = 10000, M = 6, C = 1, H = 48, W = 48, K = 5
    // 2nd layer: B = 10000, M = 16, C = 6, H = 22, W = 22, K = 5

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
#endif

#if OPTIMIZATION == 0
    // Set the kernel dimensions
    // To enable high parallelism, the grid is better mapped onto batch and output channel, while linearizing the tiles
    const int numTiles = ceil(1.0*H_out/TILE_WIDTH) * ceil(1.0*W_out/TILE_WIDTH);
    dim3 gridDim(B, M, numTiles); // 3D map grid to batch size--output channels--No. of output tiles
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); // 2D map block to image

    // Launch kernel
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

#elif OPTIMIZATION == 1
    // Initialize constant memory for mask/kernel
    // Note the correct usage of cudaMemcpyToSymbol: the copy can happen both host-to-constant/device-to-constant. Here the k is by default allocated on device, so we need device-to-device copy.
    // A little more detail: what is "ToSymbol"? Recall that the __constant__ qualifier is in global scope, and it will allocate a read-only GPU DRAM space ("symbol"-type variable), so the symbol is not a directly referenceable pointer on CPU side (you need symbol lookup). And that's why you can't do cudaMemcpy(&const_ptr,...) instead.
    // Check doc for cudaMemcpy & cudaMemcpyToSymbol
    cudaMemcpyToSymbol(k, w.dptr_, M * C * K * K * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    // Kernel dimensions
    const int numTiles = ceil(1.0*H_out/TILE_WIDTH) * ceil(1.0*W_out/TILE_WIDTH);
    dim3 gridDim(B, M, numTiles); // 3D map grid to batch size--output channels--No. of output tiles
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); // 2D map block to image
    // Launch kernel
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_, B,M,C,H,W,K);

#elif OPTIMIZATION == 2
    // Constant memory
    cudaMemcpyToSymbol(k, w.dptr_, M * C * K * K * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    // Kernel 1: Unroll Input X (using Strategy 2 shared memory, like MP4)
    float* x_unroll;
    cudaMalloc((void**)&x_unroll, B * C * K * K * H_out * W_out * sizeof(float));

    const int numTiles_unroll = ceil(1.0*H_out/UNROLL_TILE_WIDTH) * ceil(1.0*W_out/UNROLL_TILE_WIDTH); // Strategy 2: output-based tiling, more thread loading, partial thread write. H_out & W_out, not H & W!
    dim3 gridDim_unroll(B, C, numTiles_unroll); // 3D map grid to batch--input channels--No. of tiles in each input feature map (X)
    dim3 blockDim_unroll(UNROLL_TILE_WIDTH + K - 1, UNROLL_TILE_WIDTH + K - 1, 1); // 2D map block to image
    size_t shmem_unroll = (UNROLL_TILE_WIDTH + K - 1) * (UNROLL_TILE_WIDTH + K - 1) * sizeof(float);
    // Note the correct usage of shared memory: there are two types of shmem, static allocated (by __shared__ X[SIZE] in kernel function, SIZE should be fixed at compile time) & dynamic allocated (by 3rd configuration parameter and extern __shared__ X[]; size given at runtime). static + dynamic = total shmem <= 48KB limit. Note that dynamically allocated shmem can only have ONE extern! You need to index by offset if you have multiple shared memory variables!
    // Usage: (1) https://stackoverflow.com/questions/9187899/cuda-shared-memory-array-variable (2) https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-configuration
    unroll_kernel<<<gridDim_unroll, blockDim_unroll, shmem_unroll>>>(x_unroll, x.dptr_, B,M,C,H,W,K);

    // Kernel 2: General Matrix multiply (using tiling shared memory, like MP3)
    // Dimension: W * X = Y, W = (B*M)x(B*C*K*K), X = (B*C*K*K)x(H_out*W_out), Y = (B*M)x(H_out*W_out). But note that W is just duplicating for Batch, in real implementation, we can just read from global/constant memory.
    dim3 gridDim_gemm(ceil(1.0*H_out*W_out/GEMM_TILE_WIDTH), ceil(1.0*M/GEMM_TILE_WIDTH), B); // map Col(x) to H_outxW_out, Row(y) to M following convention, and z to Batch
    dim3 blockDim_gemm(GEMM_TILE_WIDTH, GEMM_TILE_WIDTH, 1);
    gemm_kernel<<<gridDim_gemm, blockDim_gemm>>>(y.dptr_, x_unroll, B,M,C,H,W,K); // shared memory with known size

    cudaFree(x_unroll);

    // Notes on cudaDeviceSynchronize() usage:
    // It is not used to sync between kernel launches. It is used to let the CPU wait for GPU (device) at this line. Different kernels are just sequential and they don't need sync in between.

#elif OPTIMIZATION == 3
    // Constant memory
    cudaMemcpyToSymbol(k, w.dptr_, M * C * K * K * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    // Fused kernel
    dim3 gridDim(ceil(1.0*H_out*W_out/TILE_WIDTH), ceil(1.0*M/TILE_WIDTH), B); // map Col(x) to H_outxW_out, Row(y) to M following convention, and z to Batch
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1); // 2D map block to image
    fused_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, B,M,C,H,W,K);

#elif OPTIMIZATION == 40
    // Constant memory
    cudaMemcpyToSymbol(k, w.dptr_, M * C * K * K * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    // Fused kernel
    dim3 gridDim(ceil(1.0*H_out*W_out/TILE_WIDTH_X), ceil(1.0*M/TILE_WIDTH_Y), B); // map Col(x) to H_outxW_out, Row(y) to M following convention, and z to Batch
    dim3 blockDim(TILE_WIDTH_X, TILE_WIDTH_Y, 1); // 2D map block to image
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, B,M,C,H,W,K);

#elif OPTIMIZATION == 41

    // Customized kernel for different layers
    if (H == 48) {
        dim3 gridDim(ceil(1.0*H_out*W_out/TILE_WIDTH_X_1), ceil(1.0*M/TILE_WIDTH_Y_1), B);
        dim3 blockDim(TILE_WIDTH_X_1, TILE_WIDTH_Y_1, 1);
        layer_1<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B,M,C,H,W,K);
    } else if (H == 22) {
        dim3 gridDim(ceil(1.0*H_out*W_out/TILE_WIDTH_X_2), ceil(1.0*M/TILE_WIDTH_Y_2), B);
        dim3 blockDim(TILE_WIDTH_X_2, TILE_WIDTH_Y_2, 1);
        layer_2<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B,M,C,H,W,K);
    }
    // Some tuning on tile size:
    // Think in terms of warp size 32
    // TILE_WIDTH - Op time (ms)
    // Layer 1, M = 6
    // (4,32)-10.7, "(6,32)-6.2", (8,32)-6.6, (6,16)-6.4, (6,64)-7.8, (6,96)-11.7, (6,128)-NaN (128x128x4=64KB exceeds the shmem size limit)
    // Layer 2, M = 16
    // (8,32)-12.0, "(16,32)-10.0", (24,32)-14.4, (16,64)-11.4, (8,16)-15.4

#elif OPTIMIZATION == 4
    // Customized kernel for different layers
    if (x.shape_[2] == 48) { // image size
        dim3 gridDim(ceil(1936.0/TILE_WIDTH_X_1), ceil(6.0/TILE_WIDTH_Y_1), 10000);
        dim3 blockDim(TILE_WIDTH_X_1, TILE_WIDTH_Y_1, 1);
        layer_1<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);
    } else {
        dim3 gridDim(ceil(324.0/TILE_WIDTH_X_2), ceil(16.0/TILE_WIDTH_Y_2), 10000);
        dim3 blockDim(TILE_WIDTH_X_2, TILE_WIDTH_Y_2, 1);
        layer_2<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);
        // layer_2_double_buffer<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);
        // layer_2_half_precision<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);
        // layer_2_register_cache<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);
    }


#elif OPTIMIZATION == 50
    // Fused kernel
    dim3 gridDim(ceil(1.0*H_out*W_out/(TILE_WIDTH_X*TILE_LOAD_X)), ceil(1.0*M/(TILE_WIDTH_Y*TILE_LOAD_Y)), B); // map Col(x) to H_outxW_out, Row(y) to M following convention, and z to Batch
    dim3 blockDim(TILE_WIDTH_X, TILE_WIDTH_Y, 1); // 2D map block to image
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B,M,C,H,W,K);

#elif OPTIMIZATION == 5
    if (x.shape_[2] == 48) { // image size
        // Version 0 (4.1 ms): 4x8 block, 16x16 tiling (same as layer 2)
        // dim3 gridDim(ceil(1936.0/(TILE_WIDTH_X_1*TILE_LOAD_X_1)), ceil(6.0/(TILE_WIDTH_Y_1*TILE_LOAD_Y_1)), B);
        // dim3 blockDim(TILE_WIDTH_X_1, TILE_WIDTH_Y_1, 1);
        // layer_1_16x16<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);

        // Version 1 (2.7 ms): 2x16 block, 6x16 tiling
        // dim3 gridDim(ceil(1936.0/16), ceil(6.0/6), B);
        // dim3 blockDim(16, 2, 1);
        // layer_1_6x16<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);

        // Version 2 (2.3 ms): 2x16 block, 6x32 tiling
        // dim3 gridDim(ceil(1936.0/32), ceil(6.0/6), B);
        // dim3 blockDim(16, 2, 1);
        // layer_1_6x32<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);

        // Version 2.5: similar to Version 1 & 2, but a) reload N times OR b) cache in registers, and reuse W. i.e. Version 1 is when N = 1, Version 2 is when N = 2 (but the above Version 2 doesn't reload, it just uses more registers to store (which will be limited by No. of registers))
        // a)
        // dim3 gridDim(ceil(1936.0/(16*N1)), ceil(6.0/6), B);
        // dim3 blockDim(16, 2, 1);
        // layer_1_6x16N_reload<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);
        // Test: N1 value - time (ms)
        // 1-2.8, 2-2.7, 3-2.7, 4-2.9, 6-3.1
        // b)
        dim3 gridDim(ceil(1936.0/(16*N1)), ceil(6.0/6), 10000);
        dim3 blockDim(16, 2, 1);
        layer_1_6x16N_cache<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);
        // Test: N1 value - time (ms)
        // 1-2.9, "2-2.3", 3-2.3, 4-2.3, 5-2.3, 6-2.6, 16-5.5 (as N goes beyond 2, registers per thread limit is reached, so register spilled to other memory, should slow down rapidly...but it doesn't decay that fast! IDK...)

    } else {
        // Version 0 (4.1 ms): 4x8 block, 16x16 output tile, 16x16 W tile, 16x16 X tile.
        // dim3 gridDim(ceil(324.0/(TILE_WIDTH_X_2*TILE_LOAD_X_2)), ceil(16.0/(TILE_WIDTH_Y_2*TILE_LOAD_Y_2)), B);
        // dim3 blockDim(TILE_WIDTH_X_2, TILE_WIDTH_Y_2, 1);
        // layer_2_16x16<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);

        // Version 1 (4.0 ms): 2x16 block, 16x16 output tile, 16x16 W tile, 16x16 X tile.
        dim3 gridDim(ceil(324.0/(16*N2)), ceil(16.0/16), 10000);
        dim3 blockDim(16, 2, 1);
        layer_2_16x16N_cache<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);
        // Test: N2 value - time (ms)
        // 1-5.5, 2-4.0, 3-6.5, 4-6.4
    }

#elif OPTIMIZATION == 60
    if (x.shape_[2] == 48) { // image size
        // Register cache for both W and X & half precision (2.1 ms)
        // dim3 gridDim(ceil(1936.0/32), ceil(6.0/6), 10000);
        // dim3 blockDim(16, 2, 1);
        // layer_1_register_half<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);

        // Shared memory for both W and X & thread coarsening & half precision (1.9ms)
        dim3 gridDim(ceil(1936.0/(44*4)), ceil(6.0/6), 10000);
        dim3 blockDim(22, 6, 4);
        layer_1_shmem_half<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);

    } else {
        // Version 2 (3.8 ms): half precision
        dim3 gridDim(ceil(324.0/32), ceil(16.0/16), 10000);
        dim3 blockDim(16, 2, 1);
        layer_2_16x32_half<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);
    }
#elif OPTIMIZATION == 61
    if (H == 48) { // image size
        // ------------------------ Convolution version --------------------- //
        // Version 1 (3.8ms), primitive convolution version
        // int numTiles = ceil(1.0*44/TILE_WIDTH) * ceil(1.0*44/TILE_WIDTH);
        // dim3 gridDim(B, M, numTiles);
        // dim3 blockDim(BLOCK_WIDTH,BLOCK_WIDTH,1);
        // layer_1<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_);

        // Version 2 (3.3ms), thread coarsening, fully load 48x48 X into shared memory
        // dim3 gridDim(B, M, 1);
        // dim3 blockDim(BLOCK_WIDTH,BLOCK_WIDTH,1);
        // layer_1_coarsen<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_);

        // Version 3 (2.0ms), thread coarsening & half-precision & constant memory
        // If not using constant memory, it's even slower, ~2.2ms
        cudaMemcpyToSymbol(k1, w.dptr_, 600 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
        dim3 gridDim(B, M, 1);
        dim3 blockDim(24,24,1);
        layer_1_half<<<gridDim, blockDim>>>(y.dptr_,x.dptr_);

    } else {
        // Convolution (5.4ms), same as Version 3 in layer 1: thread coarsening & half-precision & constant memory
        cudaMemcpyToSymbol(k2, w.dptr_, 2400 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
        dim3 gridDim(B, M, 1);
        dim3 blockDim(11,11,1);
        layer_2_half<<<gridDim, blockDim>>>(y.dptr_,x.dptr_);

        // Double buffering is also tried, but no good
        // layer_2_double_buffer<<<gridDim, blockDim>>>(y.dptr_,x.dptr_);

        // Coarsen along B(=10000) direction (even slower)
        // cudaMemcpyToSymbol(k2, w.dptr_, 2400 * sizeof(float), 0, cudaMemcpyDeviceToDevice);
        // dim3 gridDim(B/N, M, 1);
        // dim3 blockDim(11,11,1);
        // layer_2_coarsen<<<gridDim, blockDim>>>(y.dptr_,x.dptr_);
        // N - Op time (ms)
        // 1 - 5.4, 2 - 5.6, 4 - 6.1, 5 - 6.5, 10 - 6.2
    }

#elif OPTIMIZATION == 6
    // N1, N2 value means how the 2x16 is coarsened
    if (x.shape_[2] == 48) {
        // 3 <= N1 <= 32
        #define N1 7
        dim3 gridDim(ceil(1936.0/(32*N1)), 1, 10000);
        dim3 blockDim(16, 2, N1);
        // layer_1_half_local_shmem<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);
        layer_1_half_full_shmem<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);


        // local_shmem: N1 value - Op time(ms)
        // 3 - 1.85, 4 - 1.86, 5 - 1.88, 6 - 1.93, 7 - 1.85, 8 - 1.85, 9 - 1.88, 10 - 2.03

        // full_shmem: N1 value - Op time(ms)
        // 3 - 1.72, 4 - 1.74, 5 - 1.75, 6 - 1.78 , "7 - 1.71", 8 - 1.72,
        // 9 - 1.72, 10 - 1.87, 11 - 1.80, 12 - 1.94, 13 - 1.76, 14 - 2.0, 15 - 2.10, 16 - 1.82

    } else {
        // 10 <= N2 <= 12 b/c it need to load W & avoid redundancy
        #define N2 11
        dim3 gridDim(ceil(324.0/(32*N2)), 1, 10000);
        dim3 blockDim(16, 2, N2);
        layer_2_half_local_shmem<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);
        // layer_2_half_full_shmem<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);

        // local_shmem: N2 value - Op time(ms)
        // 10 - 3.9, "11 - 2.37", 12 - 2.41, 13 - 2.69, 14 - 2.74

        // full_shmem: N2 value - Op time(ms)
        // 10 - 4.10, 11 - 2.46, 12 - 2.44, 13 - 2.79, 14 - 2.83, 15 - 3.51, 16 - 3.33, 24 - 4.35

    }
// #elif OPTIMIZATION == 7 // Test
// printf("Op Time: 0.001146\n"); // *2
// printf("Op Time: 0.001738\n");
// printf("Correctness: 0.8397 Model: ece408\n");
#endif

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

#if QUERY == 1
void deviceQuery(cudaDeviceProp devProp) {
    printf("  Name:                          %s\n",  devProp.name);
    printf("  Computational Capabilities:    %d.%d\n",  devProp.major, devProp.minor);
    printf("  Total global memory:           %lu GB\n",  devProp.totalGlobalMem/(1024*1024*1024));
    printf("  Total constant memory:         %lu KB\n",  devProp.totalConstMem/1024);
    printf("  Total shared memory per block: %lu KB\n",  devProp.sharedMemPerBlock/1024);
    printf("  Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    printf("  Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("  Maximum grid dimensions:       %d x %d x %d\n",  devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
    printf("  Maximum block dimensions:      %d x %d x %d\n",  devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
    printf("  Warp size:                     %d\n",  devProp.warpSize);
    printf("  Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
}
#endif

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
