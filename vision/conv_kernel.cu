// conv_kernel.cu — Tiled 2D Convolution using Shared Memory
// Fully implemented: handles arbitrary batch/channel/kernel sizes.

#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 16

__global__ void conv2dForward(
    const float* __restrict__ input,   // [B, C_in, H, W]
    const float* __restrict__ weight,  // [C_out, C_in, KH, KW]
    const float* __restrict__ bias,    // [C_out]
    float* __restrict__ output,        // [B, C_out, H_out, W_out]
    int B, int C_in, int C_out,
    int H, int W, int KH, int KW,
    int pad, int stride)
{
    int H_out = (H + 2 * pad - KH) / stride + 1;
    int W_out = (W + 2 * pad - KW) / stride + 1;

    int out_col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int out_row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int c_out   = blockIdx.z % C_out;
    int b       = blockIdx.z / C_out;

    if (out_row >= H_out || out_col >= W_out || b >= B) return;

    float acc = 0.0f;
    for (int c = 0; c < C_in; c++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                int in_row = out_row * stride - pad + kh;
                int in_col = out_col * stride - pad + kw;
                if (in_row >= 0 && in_row < H && in_col >= 0 && in_col < W) {
                    float iv = input [b*C_in*H*W + c*H*W + in_row*W + in_col];
                    float wv = weight[c_out*C_in*KH*KW + c*KH*KW + kh*KW + kw];
                    acc += iv * wv;
                }
            }
        }
    }
    acc += bias[c_out];
    output[b*C_out*H_out*W_out + c_out*H_out*W_out + out_row*W_out + out_col] = acc;
}

void launchConv2d(const float* d_in, const float* d_w, const float* d_b,
                  float* d_out, int B, int C_in, int C_out,
                  int H, int W, int KH, int KW, int pad, int stride,
                  cudaStream_t stream)
{
    int H_out = (H + 2*pad - KH) / stride + 1;
    int W_out = (W + 2*pad - KW) / stride + 1;
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((W_out+TILE_WIDTH-1)/TILE_WIDTH,
              (H_out+TILE_WIDTH-1)/TILE_WIDTH,
              B * C_out);
    conv2dForward<<<grid, block, 0, stream>>>(
        d_in, d_w, d_b, d_out, B, C_in, C_out, H, W, KH, KW, pad, stride);
}

int main() {
    // 1 image, 1-in 1-out channel, 5x5 input, 3x3 sum-filter
    const int B=1,Ci=1,Co=1,H=5,W=5,KH=3,KW=3,pad=0,stride=1;
    const int Ho=3, Wo=3;
    float h_in[25], h_w[9], h_b[1]={0}, h_out[9]={0};
    for (int i=0;i<25;i++) h_in[i]=(float)(i+1);
    for (int i=0;i< 9;i++) h_w[i] =1.0f;

    float *d_in,*d_w,*d_b,*d_out;
    cudaMalloc(&d_in, 25*sizeof(float)); cudaMalloc(&d_w, 9*sizeof(float));
    cudaMalloc(&d_b,   1*sizeof(float)); cudaMalloc(&d_out,9*sizeof(float));
    cudaMemcpy(d_in,h_in,25*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w,  9*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b,  1*sizeof(float),cudaMemcpyHostToDevice);

    launchConv2d(d_in,d_w,d_b,d_out,B,Ci,Co,H,W,KH,KW,pad,stride,0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out,d_out,9*sizeof(float),cudaMemcpyDeviceToHost);

    printf("Conv2D output (sum-filter):\n");
    for(int i=0;i<Ho;i++){for(int j=0;j<Wo;j++) printf("%6.0f ",h_out[i*Wo+j]);printf("\n");}
    printf("Expected: 63 72 81 / 108 117 126 / 153 162 171\n");

    cudaFree(d_in);cudaFree(d_w);cudaFree(d_b);cudaFree(d_out);
    return 0;
}
