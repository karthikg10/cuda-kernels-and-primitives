from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_ext",
    ext_modules=[
        CUDAExtension(
            name="custom_ext_cuda",
            sources=["csrc/activations.cu", "csrc/bindings.cpp"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
