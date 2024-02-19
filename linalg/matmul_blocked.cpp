// matmul_blocked.cpp — Cache-Blocked Matrix Multiplication
// Research / Exploratory

#include <algorithm>
#include <cstring>
#include "linalg.hpp"

// Block sizes tuned for L2/L3 cache (adjust per architecture)
#ifndef BLOCK_M
#define BLOCK_M 64
#endif
#ifndef BLOCK_N
#define BLOCK_N 256
#endif
#ifndef BLOCK_K
#define BLOCK_K 64
#endif

void matmul_blocked(const float* A, const float* B, float* C, int M, int N, int K)
{
    memset(C, 0, M * N * sizeof(float));

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; i += BLOCK_M) {
        for (int j = 0; j < N; j += BLOCK_N) {
            for (int k = 0; k < K; k += BLOCK_K) {
                int ib = std::min(BLOCK_M, M - i);
                int jb = std::min(BLOCK_N, N - j);
                int kb = std::min(BLOCK_K, K - k);

                // Inner block computation (fits in L1/L2)
                for (int ii = 0; ii < ib; ii++) {
                    for (int kk = 0; kk < kb; kk++) {
                        float a = A[(i + ii) * K + (k + kk)];
                        for (int jj = 0; jj < jb; jj++) {
                            C[(i + ii) * N + (j + jj)] +=
                                a * B[(k + kk) * N + (j + jj)];
                        }
                    }
                }
            }
        }
    }
}
