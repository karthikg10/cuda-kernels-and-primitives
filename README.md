# CUDA Kernels & Primitives
> 🔬 **Research / Exploratory** — A collection of hand-tuned GPU and CPU compute kernels spanning vision, linear algebra, custom PyTorch ops, and recommendation systems.

This repo demonstrates low-level performance engineering across four domains — each subfolder is a self-contained project sharing the common theme of **squeezing peak hardware throughput** through CUDA, AVX2, and OpenMP.

---

## Repository Structure

```
cuda-kernels-and-primitives/
├── vision/          ← Parallel CNN forward-pass (conv + pooling CUDA kernels)
├── linalg/          ← High-performance matrix multiplication (AVX2 + OpenMP)
├── ops/             ← Custom PyTorch C++ / CUDA extension (activation functions)
└── recommender/     ← GPU-parallelized matrix factorization SGD
```

---

## Projects

### 🔷 `vision/` — Parallel CNN Image Classifier
Hand-written CUDA kernels for CNN forward-pass replacing OpenCV CPU baseline.

**Key techniques:** Shared memory tiling · Coalesced memory access · CUDA streams

| Implementation | Latency (ms) | Speedup |
|---|---|---|
| CPU (OpenCV baseline) | 84.3 | 1.0× |
| Naive CUDA | 71.2 | 1.18× |
| Shared Mem + Coalesced | 49.8 | 1.69× |
| + CUDA Streams | **47.5** | **1.77×** |

---

### 🔷 `linalg/` — High-Performance Linear Algebra Engine
Cache-blocked GEMM with AVX2 SIMD micro-kernels and OpenMP threading.

**Key techniques:** AVX2 FMA (8-wide) · L1/L2/L3 cache blocking · False-sharing avoidance

```
Cache Blocking Strategy:
  L1 micro-kernel:  8×8   (fits 32KB L1)
  L2 panel:        64×256  (fits 256KB L2)
  L3 block:       512×512  (fits 8MB L3)
```

| Matrix Size | NumPy (MKL) | Naive C++ | OpenMP + AVX2 |
|---|---|---|---|
| 512×512 | 2.1ms | 18.4ms | **2.6ms** |
| 2048×2048 | 89ms | 1,184ms | **97ms** |
| 4096×4096 | 680ms | 9,500ms | **725ms** |

> Within ~8% of Intel MKL at large matrix sizes.

---

### 🔷 `ops/` — Custom C++ & CUDA Extensions for PyTorch
pybind11-bound CUDA kernels for non-standard activations (Swish, Mish, fused bias+act), bypassing Python interpreter overhead.

**Key techniques:** `float4` vectorized loads · Near-peak memory BW · pybind11 torch extension

| Op | PyTorch Native | Custom CUDA | Speedup |
|---|---|---|---|
| Swish (BS=1) | 1.82ms | 0.44ms | **4.1×** |
| Mish (BS=1) | 2.41ms | 0.61ms | **3.95×** |
| Fused Bias+Swish (BS=1) | 3.12ms | 0.72ms | **4.3×** |

```bash
# Build and install
cd ops && pip install -e .
python bench_activations.py
```

---

### 🔷 `recommender/` — GPU-Parallelized Matrix Factorization
CUDA-based SGD with memory tiling for large-scale collaborative filtering.

**Key techniques:** Parallel SGD updates · Memory tiling · L2 regularization

| Method | Training Time | RMSE |
|---|---|---|
| CPU SGD (sklearn) | 118s | 0.889 |
| CUDA SGD (naive) | 98s | 0.891 |
| CUDA SGD + Tiling | **77s** | **0.888** |

> 35% runtime reduction vs CPU baseline on MovieLens-1M.

---

## Build (CUDA Projects)

```bash
# Requires: CUDA 12+, GCC 11+, OpenMP
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## Build (PyTorch Extension)
```bash
cd ops && pip install -e .
python -c "import custom_ext_cuda; print('OK')"
```

## Requirements
```
CUDA >= 12.0
GCC >= 11.0 with AVX2 support
PyTorch >= 2.1
OpenCV >= 4.0 (vision module)
```
