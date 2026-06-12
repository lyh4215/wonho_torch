# setup.py

from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "wonho_torch.backend._C",
        ["wonho_torch/backend/native_ops.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    )
]

setup(
    name="wonho_torch",
    version="0.0.1",
    packages=["wonho_torch", "wonho_torch.backend"],
    ext_modules=ext_modules,
)