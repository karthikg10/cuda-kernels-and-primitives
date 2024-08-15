// bindings.cpp — pybind11 / torch::extension bindings
// Research / Exploratory

#include <torch/extension.h>

// Forward declarations (implemented in activations.cu)
void launchSwish(const float* x, float* y, int N, cudaStream_t s);
void launchMish(const float* x, float* y, int N, cudaStream_t s);

torch::Tensor swish_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    auto y = torch::empty_like(x);
    // Use the current PyTorch CUDA stream for kernel launch
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launchSwish(x.data_ptr<float>(), y.data_ptr<float>(), x.numel(), stream);
    return y;
}

torch::Tensor mish_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    auto y = torch::empty_like(x);
    launchMish(x.data_ptr<float>(), y.data_ptr<float>(), x.numel(), 0);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swish", &swish_cuda, "Swish activation (CUDA)");
    m.def("mish",  &mish_cuda,  "Mish activation (CUDA)");
}
