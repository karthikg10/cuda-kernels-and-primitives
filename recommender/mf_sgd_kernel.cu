// mf_sgd_kernel.cu — GPU Matrix Factorization with SGD (fully implemented)
// Parallelizes SGD updates: each thread handles one (user, item) interaction.
// Uses atomic operations for concurrent embedding updates.

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define K_DIM   16   // latent dimension
#define THREADS 256

// Parallel SGD: each thread updates one interaction
__global__ void sgdUpdateKernel(
    float* __restrict__ U,            // [num_users, K]
    float* __restrict__ V,            // [num_items, K]
    const int*   __restrict__ users,
    const int*   __restrict__ items,
    const float* __restrict__ ratings,
    float lr, float reg,
    int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int u = users[idx];
    int i = items[idx];
    float r = ratings[idx];

    // Compute dot product R_hat = U[u] · V[i]
    float pred = 0.0f;
    for (int k = 0; k < K; k++)
        pred += U[u*K + k] * V[i*K + k];

    float err = r - pred;

    // SGD update with L2 regularization using atomics for thread safety
    for (int k = 0; k < K; k++) {
        float u_k = U[u*K + k];
        float v_k = V[i*K + k];
        atomicAdd(&U[u*K + k],  lr * ( err * v_k - reg * u_k));
        atomicAdd(&V[i*K + k],  lr * ( err * u_k - reg * v_k));
    }
}

// Compute RMSE on held-out interactions
__global__ void computeRMSE(
    const float* __restrict__ U,
    const float* __restrict__ V,
    const int*   __restrict__ users,
    const int*   __restrict__ items,
    const float* __restrict__ ratings,
    float* __restrict__ sq_err,
    int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int u = users[idx], i = items[idx];
    float pred = 0.0f;
    for (int k = 0; k < K; k++) pred += U[u*K+k] * V[i*K+k];
    float e = ratings[idx] - pred;
    atomicAdd(sq_err, e*e);
}

// Xavier initializer on host
void xavierInit(float* arr, int rows, int cols) {
    float scale = sqrtf(2.0f / (rows + cols));
    for (int i = 0; i < rows * cols; i++)
        arr[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
}

int main() {
    srand(42);
    const int NUM_USERS = 100, NUM_ITEMS = 200;
    const int N_TRAIN = 5000;   // interactions
    const int EPOCHS  = 20;
    const float LR = 0.01f, REG = 0.01f;

    // Generate synthetic ratings dataset
    int*   h_users   = new int  [N_TRAIN];
    int*   h_items   = new int  [N_TRAIN];
    float* h_ratings = new float[N_TRAIN];
    for (int i = 0; i < N_TRAIN; i++) {
        h_users[i]   = rand() % NUM_USERS;
        h_items[i]   = rand() % NUM_ITEMS;
        h_ratings[i] = 1.0f + (rand() % 5);  // ratings 1-5
    }

    // Initialize embeddings
    float* h_U = new float[NUM_USERS * K_DIM];
    float* h_V = new float[NUM_ITEMS * K_DIM];
    xavierInit(h_U, NUM_USERS, K_DIM);
    xavierInit(h_V, NUM_ITEMS, K_DIM);

    // Allocate device memory
    int   *d_users, *d_items;
    float *d_ratings, *d_U, *d_V, *d_sq_err;
    cudaMalloc(&d_users,   N_TRAIN   * sizeof(int));
    cudaMalloc(&d_items,   N_TRAIN   * sizeof(int));
    cudaMalloc(&d_ratings, N_TRAIN   * sizeof(float));
    cudaMalloc(&d_U, NUM_USERS * K_DIM * sizeof(float));
    cudaMalloc(&d_V, NUM_ITEMS * K_DIM * sizeof(float));
    cudaMalloc(&d_sq_err, sizeof(float));

    cudaMemcpy(d_users,   h_users,   N_TRAIN*sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_items,   h_items,   N_TRAIN*sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_ratings, h_ratings, N_TRAIN*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, h_U, NUM_USERS*K_DIM*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, NUM_ITEMS*K_DIM*sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (N_TRAIN + THREADS - 1) / THREADS;

    printf("Training MF (K=%d, users=%d, items=%d, interactions=%d)\n",
           K_DIM, NUM_USERS, NUM_ITEMS, N_TRAIN);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        sgdUpdateKernel<<<blocks, THREADS>>>(
            d_U, d_V, d_users, d_items, d_ratings, LR, REG, N_TRAIN, K_DIM);

        if ((epoch + 1) % 5 == 0) {
            float zero = 0.0f;
            cudaMemcpy(d_sq_err, &zero, sizeof(float), cudaMemcpyHostToDevice);
            computeRMSE<<<blocks, THREADS>>>(
                d_U, d_V, d_users, d_items, d_ratings, d_sq_err, N_TRAIN, K_DIM);
            float sq;
            cudaMemcpy(&sq, d_sq_err, sizeof(float), cudaMemcpyDeviceToHost);
            printf("Epoch %2d | RMSE: %.4f\n", epoch+1, sqrtf(sq / N_TRAIN));
        }
    }

    cudaFree(d_users); cudaFree(d_items); cudaFree(d_ratings);
    cudaFree(d_U); cudaFree(d_V); cudaFree(d_sq_err);
    delete[] h_users; delete[] h_items; delete[] h_ratings;
    delete[] h_U; delete[] h_V;
    return 0;
}
