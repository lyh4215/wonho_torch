#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "cuda_array.h"

namespace py = pybind11;

py::array_t<double> add_forward(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b
);

std::shared_ptr<CudaArray> add_storage(
    std::shared_ptr<CudaArray> a,
    std::shared_ptr<CudaArray> b
);