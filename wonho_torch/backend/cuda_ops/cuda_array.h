#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <memory>
#include <string>

namespace py = pybind11;

class CudaArray {
public:
    double* ptr;
    std::vector<ssize_t> shape;
    size_t size;

    explicit CudaArray(const std::vector<ssize_t>& shape_);
    ~CudaArray();

    CudaArray(const CudaArray&) = delete;
    CudaArray& operator=(const CudaArray&) = delete;

    static std::shared_ptr<CudaArray> from_numpy(
        py::array_t<double, py::array::c_style | py::array::forcecast> arr
    );

    py::array_t<double> to_numpy() const;

    std::string repr() const;
};