
#ifndef MXNET_OPERATOR_NEW_FORWARD_H_
#define MXNET_OPERATOR_NEW_FORWARD_H_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{


template <typename cpu, typename DType>
void forward(mshadow::Tensor<cpu, 4, DType> &y, const mshadow::Tensor<cpu, 4, DType> &x, const mshadow::Tensor<cpu, 4, DType> &k)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    The code in 16 is for a single image.
    We have added an additional dimension to the tensors to support an entire mini-b
    The goal here is to be correct, not fast (this is the CPU implementation.)
    */

    /*
    Data layout:
    y: output data, b size * output channels * y * x
    x: input data, b size * input channels * y * x
    k: kernel weights, output channels * input channels * y * x
    */

    const int B = x.shape_[0]; // b size
    const int M = y.shape_[1]; // output channels
    const int C = x.shape_[1]; // input channels
    const int H = x.shape_[2]; // image height
    const int W = x.shape_[3]; // image width
    const int K = k.shape_[3]; // kernel size

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define UNROLL 1
// 0: plain CPU version
// 1: Unroll and Matrix Multiply CPU version

#if UNROLL == 0

    for (int b = 0; b < B; ++b) { // for each image in the b

        // CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";

        /* ... a bunch of nested loops later...
            y[b][m][h][w] += x[b][c][h + p][w + q] * k[m][c][p][q];
        */

        for (int m = 0; m < M; m++) { // for each output feature maps
            for (int h = 0; h < H_out; h++) { // for each output element
                for (int w = 0; w < W_out; w++) {
                    float result = 0;
                    for (int c = 0; c < C; c++) { // sum over all input feature maps
                        for (int p = 0; p < K; p++) { // for each kernel weight
                            for (int q = 0; q < K; q++) {
                                result += x[b][c][h+p][w+q] * k[m][c][p][q];
                                // Unlike the convolutions described in the class, note that this one is NOT centered on the input image. It's a forward kernel rather than center kernel. Note H-K+1 and W-K+1!
                            }
                        }
                    }
                    y[b][m][h][w] = result;
                }
            }
        }
    }

#elif UNROLL == 1
    // Memcpy to CPU space (use cudaMemcpy if this code is in .cuh file)
    float* x_cpu = (float*)malloc(B * C * H * W * sizeof(float));
    float* k_cpu = (float*)malloc(M * C * K * K * sizeof(float));
    memcpy(x_cpu, x.dptr_, B * C * H * W * sizeof(float));
    memcpy(k_cpu, k.dptr_, M * C * K * K * sizeof(float));
    // cudaMemcpy(x_cpu, x.dptr_, B * C * H * W * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(k_cpu, w.dptr_, M * C * K * K * sizeof(float), cudaMemcpyDeviceToHost);

    float* y_cpu = (float*)malloc(B * M * H_out * W_out * sizeof(float));
    float* x_unroll_cpu = (float*)malloc(B * C * K * K * H_out * W_out * sizeof(float));

#define x4d(i3, i2, i1, i0) x_cpu[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * W + i0]
#define k4d(i3, i2, i1, i0) k_cpu[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define y4d(i3, i2, i1, i0) y_cpu[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * W_out + i0]
#define x_unroll4d(i3, i2, i1, i0) x_unroll_cpu[(i3) * (C * K * K * H_out * W_out) + (i2) * (K * K * H_out * W_out) + (i1) * (H_out * W_out) + i0]

    // Unroll
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    for (int p = 0; p < K; p++) {
                        for (int q = 0; q < K; q++) {
                            x_unroll4d(b, c, p*K+q, h*W_out+w) = x4d(b, c, h+p, w+q);
                        }
                    }
                }
            }
        }
    }

    // Matrix Multiply
    for (int b = 0; b < B; b++) {
        int offset_y = b * M * H_out * W_out;
        int offset_x = b * C * K * K * H_out * W_out;
        for (int m = 0; m < M; m++) {
            for (int hw = 0; hw < H_out * W_out; hw++) { // h & w linearized
                float result = 0; // int result = 0; FUCK!!!
                for (int t = 0; t < C * K * K; t++) {
                    result += k_cpu[m * (C * K * K) + t] * x_unroll_cpu[offset_x + t * (H_out * W_out) + hw];
                }
                y_cpu[offset_y + m * H_out * W_out + hw] = result;
            }
        }
    }

    memcpy(y.dptr_, y_cpu, B * M * H_out * W_out * sizeof(float));
    // cudaMemcpy(y.dptr_, y_cpu, B*M*H_out*W_out*sizeof(float), cudaMemcpyHostToDevice);

#undef x4d
#undef k4d
#undef y4d
#undef x_unroll4d

#endif

}
}
}



#endif
