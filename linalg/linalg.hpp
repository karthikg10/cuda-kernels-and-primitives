#pragma once
// linalg.hpp — Public API for Linear Algebra Engine

// Matrix multiplication variants
void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K);
void matmul_omp(const float* A, const float* B, float* C, int M, int N, int K);
void matmul_avx2(const float* A, const float* B, float* C, int M, int N, int K);
void matmul_blocked(const float* A, const float* B, float* C, int M, int N, int K);
