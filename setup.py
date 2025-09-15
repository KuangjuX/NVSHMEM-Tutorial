import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (  # pylint: disable=import-error
    BuildExtension,
    CUDAExtension,
)


def get_cuda_root():
    # get cuda root by nvcc executable path
    nvcc = os.popen("which nvcc").read().strip()
    cuda_root = os.path.dirname(os.path.dirname(nvcc))
    return os.path.abspath(cuda_root)


cuda_root = get_cuda_root()
nvshmem_home = os.environ.get("NVSHMEM_HOME", "/opt/nvshmem")
current_dir = os.path.dirname(os.path.abspath(__file__))

include_dirs = [
    os.path.join(current_dir, "csrc"),  # Add csrc directory to include path
    os.path.join(nvshmem_home, "include"),
    os.path.join(cuda_root, "include"),
]

library_dirs = [
    os.path.join(nvshmem_home, "lib"),
    "/lib/x86_64-linux-gnu/",
]

nvcc_dlink = ["-dlink", f"-L{nvshmem_home}/lib", "-lnvshmem_device"]
extra_link_args = [
    f"-l:libnvshmem_host.so",
    "-l:libnvshmem_device.a",
    f"-Wl,-rpath,{nvshmem_home}/lib",
    f"-L/lib/x86_64-linux-gnu/",
    "-lcuda",  # Add this line to link CUDA driver library
    "-lcudart",
]

setup(
    name="nvshmem_tutorial",
    version="0.1.0",
    packages=find_packages(exclude=["benchmarks"]),
    ext_modules=[
        CUDAExtension(
            name="_nvshmem_tutorial",
            sources=[
                "csrc/buffer.cu",
                "csrc/intranode.cu",
                "csrc/internode.cu",
                "csrc/pybind.cu",
                "csrc/kernels/copy.cu",
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_compile_args={
                "cxx": [
                    "-std=c++20",
                    "-O3",
                ],
                "nvcc": [
                    "-O3",
                    "-gencode",
                    "arch=compute_90a,code=sm_90a",
                    "-DKITTENS_HOPPER",
                    "-rdc=true",
                    "-std=c++20",
                    "--keep",
                    "--generate-line-info",
                    "-Wno-deprecated-gpu-targets",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                ],
                "nvcc_dlink": nvcc_dlink,
            },
            extra_link_args=extra_link_args,
            libraries=["cuda"],
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
    install_requires=[
        "torch",
    ],
)
