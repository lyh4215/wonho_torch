# setup.py

import os
import shutil
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

import pybind11
import numpy as np


def find_cuda_home():
    # 1. 환경변수 우선
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is not None:
        nvcc = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.exists(nvcc):
            return cuda_home

    # 2. PATH에서 nvcc 찾기
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is not None:
        # /usr/local/cuda/bin/nvcc -> /usr/local/cuda
        return os.path.dirname(os.path.dirname(nvcc_path))

    # 3. 흔한 기본 경로
    default_cuda = "/usr/local/cuda"
    default_nvcc = os.path.join(default_cuda, "bin", "nvcc")
    if os.path.exists(default_nvcc):
        return default_cuda

    return None


CUDA_HOME = find_cuda_home()
FORCE_CUDA = os.environ.get("WONHO_TORCH_FORCE_CUDA") == "1"

HAS_CUDA = CUDA_HOME is not None


class BuildExt(build_ext):
    def build_extensions(self):
        # .cu 파일을 컴파일 대상으로 인식하게 추가
        self.compiler.src_extensions.append(".cu")

        original_compile = self.compiler._compile

        def custom_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            src_ext = os.path.splitext(src)[1]

            if isinstance(extra_postargs, dict):
                if src_ext == ".cu":
                    postargs = extra_postargs.get("nvcc", [])
                else:
                    postargs = extra_postargs.get("cxx", [])
            else:
                postargs = extra_postargs

            if src_ext == ".cu":
                if CUDA_HOME is None:
                    raise RuntimeError(
                        "CUDA_HOME was not found, but a .cu file is being compiled."
                    )

                nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
                original_compiler_so = self.compiler.compiler_so

                self.compiler.set_executable("compiler_so", nvcc)

                try:
                    original_compile(obj, src, ext, cc_args, postargs, pp_opts)
                finally:
                    self.compiler.set_executable(
                        "compiler_so",
                        original_compiler_so,
                    )
            else:
                original_compile(obj, src, ext, cc_args, postargs, pp_opts)

        self.compiler._compile = custom_compile
        super().build_extensions()


include_dirs = [
    pybind11.get_include(),
    np.get_include(),
]


ext_modules = [
    Extension(
        "wonho_torch.backend._C",
        ["wonho_torch/backend/native_ops.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
        },
    )
]


if HAS_CUDA:
    print(f"Building CUDA extension with CUDA_HOME={CUDA_HOME}")

    ext_modules.append(
        Extension(
            "wonho_torch.backend._CUDA",
            ["wonho_torch/backend/cuda_ops.cu"],
            include_dirs=[
                *include_dirs,
                os.path.join(CUDA_HOME, "include"),
            ],
            library_dirs=[
                os.path.join(CUDA_HOME, "lib64"),
            ],
            libraries=["cudart"],
            language="c++",
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--std=c++17",
                    "-Xcompiler",
                    "-fPIC",
                ],
            },
        )
    )
else:
    message = "CUDA_HOME/nvcc not found. Skipping CUDA extension."

    if FORCE_CUDA:
        raise RuntimeError(message)

    print(message)


setup(
    name="wonho_torch",
    version="0.0.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExt,
    },
)