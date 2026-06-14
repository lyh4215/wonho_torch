#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "cuda_array.h"

namespace py = pybind11;

py::array_t<double> matmul_forward_naive(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b
);

py::array_t<double> matmul_forward_tiled(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b
);

py::array_t<double> matmul_forward_cublas(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b
);

std::shared_ptr<CudaArray> matmul_storage_cublas(
    std::shared_ptr<CudaArray> a,
    std::shared_ptr<CudaArray> b
);