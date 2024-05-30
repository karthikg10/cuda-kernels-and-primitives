// main.cpp — CNN Inference Entry Point (conv + pool + FC)
// Wires together conv_kernel, pool_kernel, and a simple FC layer.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Declared in their respective .cu files
extern void launchConv2d(const float*, const float*, const float*, float*,
                         int, int, int, int, int, int, int, int, int, cudaStream_t);
extern void launchMaxpool2d(const float*, float*, int, int, int, int,
                            int, int, int, cudaStream_t);

// Tiny CNN: Conv(1->4, 3x3) -> MaxPool(2x2) -> FC -> 10 classes
void runCNNInference(int batch_size, bool benchmark) {
    const int Ci=1, Co=4, H=28, W=28, KH=3, KW=3, pad=1, stride=1;
    const int Ho=28, Wo=28;           // same-padding
    const int Hp=14, Wp=14;           // after 2x2 pool
    const int flat = Co * Hp * Wp;   // 4*14*14 = 784
    const int num_classes = 10;

    size_t in_sz    = batch_size * Ci * H * W;
    size_t conv_sz  = batch_size * Co * Ho * Wo;
    size_t pool_sz  = batch_size * Co * Hp * Wp;

    float *d_in, *d_w, *d_b, *d_conv, *d_pool, *d_fc_w, *d_out;
    cudaMalloc(&d_in,   in_sz   * sizeof(float));
    cudaMalloc(&d_w,    Co*Ci*KH*KW * sizeof(float));
    cudaMalloc(&d_b,    Co * sizeof(float));
    cudaMalloc(&d_conv, conv_sz * sizeof(float));
    cudaMalloc(&d_pool, pool_sz * sizeof(float));
    cudaMalloc(&d_fc_w, flat * num_classes * sizeof(float));
    cudaMalloc(&d_out,  batch_size * num_classes * sizeof(float));

    // Initialize with random values
    float *h_tmp = new float[Co*Ci*KH*KW];
    for (int i=0; i<Co*Ci*KH*KW; i++) h_tmp[i] = ((float)rand()/RAND_MAX - 0.5f)*0.1f;
    cudaMemcpy(d_w, h_tmp, Co*Ci*KH*KW*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_b, 0, Co*sizeof(float));
    cudaMemset(d_in, 0, in_sz*sizeof(float));
    delete[] h_tmp;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    int iters = benchmark ? 100 : 1;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, stream);

    for (int it = 0; it < iters; it++) {
        // Conv layer
        launchConv2d(d_in, d_w, d_b, d_conv,
                     batch_size, Ci, Co, H, W, KH, KW, pad, stride, stream);
        // MaxPool
        launchMaxpool2d(d_conv, d_pool, batch_size, Co, Ho, Wo, 2, 2, 2, stream);
        // FC: [B, flat] x [flat, 10] -> [B, 10]  via cuBLAS
        float alpha=1.f, beta=0.f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    num_classes, batch_size, flat,
                    &alpha, d_fc_w, num_classes,
                            d_pool, flat,
                    &beta,  d_out,  num_classes);
    }
    cudaEventRecord(t1, stream);
    cudaStreamSynchronize(stream);

    if (benchmark) {
        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        printf("[Benchmark] BS=%d | Avg latency: %.3f ms | Throughput: %.1f img/s\n",
               batch_size, ms/iters, batch_size * iters / (ms/1000.0f));
    } else {
        printf("[CNN] Forward pass complete. Output shape: [%d, %d]\n",
               batch_size, num_classes);
    }

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(d_in); cudaFree(d_w); cudaFree(d_b);
    cudaFree(d_conv); cudaFree(d_pool); cudaFree(d_fc_w); cudaFree(d_out);
}

int main(int argc, char** argv) {
    int batch_size = 32;
    bool benchmark = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--batch-size" && i+1 < argc)
            batch_size = atoi(argv[++i]);
        if (std::string(argv[i]) == "--benchmark")
            benchmark = true;
    }
    srand(42);
    runCNNInference(batch_size, benchmark);
    return 0;
}
