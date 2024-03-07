// matmul_avx2.cpp — AVX2 SIMD Micro-kernel
// Research / Exploratory

#include <immintrin.h>   // AVX2
#include <omp.h>
#include <cstring>
#include <algorithm>
#include "linalg.hpp"

// 8-wide FMA micro-kernel: computes C[i:i+8][j] += A[i:i+8][k] * B[k][j]
static void microkernel_8x1(const float* A, const float* B, float* C,
                              int M, int N, int K, int row_base, int col)
{
    __m256 c_vec = _mm256_setzero_ps();

    for (int k = 0; k < K; k++) {
        __m256 a_vec = _mm256_loadu_ps(&A[row_base * K + k]);  // 8 elements
        __m256 b_val = _mm256_set1_ps(B[k * N + col]);         // broadcast scalar
        c_vec = _mm256_fmadd_ps(a_vec, b_val, c_vec);          // FMA
    }

    // Accumulate into C (partial sum — boundary safe via masking)
    // Ensure we don't write past the last row when M % 8 != 0
    int remaining = M - row_base;
    if (remaining >= 8) {
        __m256 old = _mm256_loadu_ps(&C[row_base * N + col]);
        _mm256_storeu_ps(&C[row_base * N + col], _mm256_add_ps(old, c_vec));
    } else {
        // Scalar fallback for partial tile at bottom of matrix
        float tmp[8]; _mm256_storeu_ps(tmp, c_vec);
        for (int r = 0; r < remaining; r++)
            C[(row_base + r) * N + col] += tmp[r];
    }
}

void matmul_avx2(const float* A, const float* B, float* C, int M, int N, int K)
{
    memset(C, 0, M * N * sizeof(float));

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; i += 8) {
        int rows = std::min(8, M - i);
        for (int j = 0; j < N; j++) {
            if (rows == 8)
                microkernel_8x1(A, B, C, M, N, K, i, j);
            else {
                // Scalar fallback for boundary rows
                for (int r = i; r < i + rows; r++) {
                    float acc = 0.0f;
                    for (int k = 0; k < K; k++)
                        acc += A[r * K + k] * B[k * N + j];
                    C[r * N + j] += acc;
                }
            }
        }
    }
}
