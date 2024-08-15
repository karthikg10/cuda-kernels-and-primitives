// activations.cu — Custom CUDA Activation Kernels (fully implemented)
// Swish, Mish, GELU (approx), and fused bias+activation.
// Uses float4 vectorized loads for near-peak memory bandwidth.

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define THREADS 256

// ── Swish: x * sigmoid(x) ───────────────────────────────────────────────────
__global__ void swishKernel(const float* __restrict__ x,
                              float* __restrict__ y, int N)
{
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base + 3 < N) {
        float4 in = reinterpret_cast<const float4*>(x)[base / 4];
        float4 out;
        out.x = in.x / (1.0f + expf(-in.x));
        out.y = in.y / (1.0f + expf(-in.y));
        out.z = in.z / (1.0f + expf(-in.z));
        out.w = in.w / (1.0f + expf(-in.w));
        reinterpret_cast<float4*>(y)[base / 4] = out;
    } else {
        // scalar tail
        for (int i = base; i < N && i < base + 4; i++)
            y[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

// ── Mish: x * tanh(softplus(x)) ─────────────────────────────────────────────
__global__ void mishKernel(const float* __restrict__ x,
                            float* __restrict__ y, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float xi = x[idx];
    y[idx] = xi * tanhf(log1pf(expf(xi)));
}

// ── GELU (fast tanh approximation) ──────────────────────────────────────────
__global__ void geluApproxKernel(const float* __restrict__ x,
                                   float* __restrict__ y, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float xi = x[idx];
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float inner = 0.7978845608f * (xi + 0.044715f * xi * xi * xi);
    y[idx] = 0.5f * xi * (1.0f + tanhf(inner));
}

// ── Fused bias addition + Swish ──────────────────────────────────────────────
__global__ void fusedBiasSwishKernel(const float* __restrict__ x,
                                      const float* __restrict__ bias,
                                      float* __restrict__ y,
                                      int N, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float val = x[idx] + bias[idx % C];
    y[idx] = val / (1.0f + expf(-val));
}

// ── Host launchers ────────────────────────────────────────────────────────────
void launchSwish(const float* x, float* y, int N, cudaStream_t s) {
    int threads = THREADS;
    int blocks  = (N / 4 + threads - 1) / threads;
    swishKernel<<<blocks, threads, 0, s>>>(x, y, N);
}

void launchMish(const float* x, float* y, int N, cudaStream_t s) {
    int blocks = (N + THREADS - 1) / THREADS;
    mishKernel<<<blocks, THREADS, 0, s>>>(x, y, N);
}

void launchGeluApprox(const float* x, float* y, int N, cudaStream_t s) {
    int blocks = (N + THREADS - 1) / THREADS;
    geluApproxKernel<<<blocks, THREADS, 0, s>>>(x, y, N);
}

void launchFusedBiasSwish(const float* x, const float* bias, float* y,
                           int N, int C, cudaStream_t s) {
    int blocks = (N + THREADS - 1) / THREADS;
    fusedBiasSwishKernel<<<blocks, THREADS, 0, s>>>(x, bias, y, N, C);
}

// ── Quick correctness test ────────────────────────────────────────────────────
int main() {
    const int N = 8;
    float h_x[N]   = {-2,-1,-0.5,0,0.5,1,2,3};
    float h_out[N] = {0};
    float *d_x, *d_out;
    cudaMalloc(&d_x,   N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);

    launchSwish(d_x, d_out, N, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);

    printf("Swish output:\n");
    for (int i = 0; i < N; i++) printf("  swish(%.1f) = %.4f\n", h_x[i], h_out[i]);

    launchMish(d_x, d_out, N, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Mish output:\n");
    for (int i = 0; i < N; i++) printf("  mish(%.1f) = %.4f\n", h_x[i], h_out[i]);

    launchGeluApprox(d_x, d_out, N, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("GELU (approx) output:\n");
    for (int i = 0; i < N; i++) printf("  gelu(%.1f) = %.4f\n", h_x[i], h_out[i]);

    cudaFree(d_x); cudaFree(d_out);
    return 0;
}
