// pool_kernel.cu — Max Pooling 2D CUDA Kernel (fully implemented)

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>

__global__ void maxpool2dForward(
    const float* __restrict__ input,  // [B, C, H, W]
    float* __restrict__ output,       // [B, C, H_out, W_out]
    int B, int C, int H, int W,
    int pool_h, int pool_w, int stride)
{
    int W_out = (W - pool_w) / stride + 1;
    int H_out = (H - pool_h) / stride + 1;

    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int bc      = blockIdx.z;  // combined b*C + c index
    int b       = bc / C;
    int c       = bc % C;

    if (out_row >= H_out || out_col >= W_out || b >= B) return;

    float max_val = -FLT_MAX;
    int r0 = out_row * stride;
    int c0 = out_col * stride;

    for (int ph = 0; ph < pool_h; ph++) {
        for (int pw = 0; pw < pool_w; pw++) {
            int idx = b*C*H*W + c*H*W + (r0+ph)*W + (c0+pw);
            if (input[idx] > max_val) max_val = input[idx];
        }
    }
    output[b*C*H_out*W_out + c*H_out*W_out + out_row*W_out + out_col] = max_val;
}

void launchMaxpool2d(const float* d_in, float* d_out,
                     int B, int C, int H, int W,
                     int pool_h, int pool_w, int stride,
                     cudaStream_t stream)
{
    int H_out = (H - pool_h) / stride + 1;
    int W_out = (W - pool_w) / stride + 1;
    dim3 block(16, 16);
    dim3 grid((W_out+15)/16, (H_out+15)/16, B*C);
    maxpool2dForward<<<grid, block, 0, stream>>>(
        d_in, d_out, B, C, H, W, pool_h, pool_w, stride);
}

int main() {
    // 1 image, 1 channel, 4x4 input, 2x2 pool, stride=2
    const int B=1,C=1,H=4,W=4,PH=2,PW=2,S=2;
    const int Ho=2,Wo=2;
    float h_in[16], h_out[4]={0};
    for (int i=0;i<16;i++) h_in[i]=(float)(i+1);

    float *d_in, *d_out;
    cudaMalloc(&d_in, 16*sizeof(float));
    cudaMalloc(&d_out, 4*sizeof(float));
    cudaMemcpy(d_in, h_in, 16*sizeof(float), cudaMemcpyHostToDevice);

    launchMaxpool2d(d_in, d_out, B, C, H, W, PH, PW, S, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 4*sizeof(float), cudaMemcpyDeviceToHost);

    printf("MaxPool2D output (2x2 pool, stride=2):\n");
    for(int i=0;i<Ho;i++){for(int j=0;j<Wo;j++) printf("%5.0f ",h_out[i*Wo+j]);printf("\n");}
    printf("Expected: 6 8 / 14 16\n");

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
