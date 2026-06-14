#include "cuda_array.h"
#include "common.cuh"

#include <sstream>
#include <stdexcept>

CudaArray::CudaArray(const std::vector<ssize_t>& shape_)
    : ptr(nullptr), shape(shape_), size(1)
{
    for (ssize_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("CudaArray: shape dimension must be positive");
        }

        size *= static_cast<size_t>(dim);
    }

    CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(double)));
}

CudaArray::~CudaArray() {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

std::shared_ptr<CudaArray> CudaArray::from_numpy(
    py::array_t<double, py::array::c_style | py::array::forcecast> arr
) {
    auto buf = arr.request();

    std::vector<ssize_t> shape;
    shape.reserve(buf.ndim);

    for (ssize_t i = 0; i < buf.ndim; ++i) {
        shape.push_back(buf.shape[i]);
    }

    auto out = std::make_shared<CudaArray>(shape);

    const double* host_ptr = static_cast<const double*>(buf.ptr);

    CUDA_CHECK(cudaMemcpy(
        out->ptr,
        host_ptr,
        out->size * sizeof(double),
        cudaMemcpyHostToDevice
    ));

    return out;
}

py::array_t<double> CudaArray::to_numpy() const {
    py::array_t<double> out(shape);
    auto out_buf = out.request();

    double* host_ptr = static_cast<double*>(out_buf.ptr);

    CUDA_CHECK(cudaMemcpy(
        host_ptr,
        ptr,
        size * sizeof(double),
        cudaMemcpyDeviceToHost
    ));

    return out;
}

std::string CudaArray::repr() const {
    std::ostringstream oss;

    oss << "CudaArray(shape=(";

    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];

        if (i + 1 < shape.size()) {
            oss << ", ";
        }
    }

    oss << "), dtype=float64, device=cuda)";

    return oss.str();
}