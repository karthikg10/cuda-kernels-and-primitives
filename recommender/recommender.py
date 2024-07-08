# recommender.py — Python wrapper for CUDA MF and sklearn baseline
# Research / Exploratory

import numpy as np
import argparse
from sklearn.decomposition import NMF
import time

def cpu_baseline(R, n_components=64, n_iter=100):
    model = NMF(n_components=n_components, max_iter=n_iter, random_state=42)
    U = model.fit_transform(R)
    V = model.components_
    return U, V

def rmse(R, U, V, mask):
    pred = U @ V
    diff = (R - pred)[mask]
    return np.sqrt(np.mean(diff ** 2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ml-1m")
    parser.add_argument("--method", default="cuda_sgd",
                        choices=["cpu_sgd", "sklearn_nmf", "cuda_sgd"])
    args = parser.parse_args()

    print(f"[Recommender] Dataset: {args.dataset}, Method: {args.method}")
    # TODO: Load MovieLens dataset
    # TODO: If cuda_sgd: call CUDA extension via ctypes or pybind
    # TODO: Evaluate RMSE and report throughput
    print("TODO: Wire up CUDA SGD kernel via ctypes/pybind11")
