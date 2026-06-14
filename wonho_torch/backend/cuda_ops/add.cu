#include "add.h"
#include "common.cuh"
#include "kernels.cuh"

#include <stdexcept>
#include <vector>

py::array_t<double> add_forward(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b
) {
    auto a_buf = a.request();
    auto b_buf = b.request();

    if (a_buf.ndim != b_buf.ndim) {
        throw std::runtime_error("cuda add_forward: ndim mismatch");
    }

    for (ssize_t i = 0; i < a_buf.ndim; ++i) {
        if (a_buf.shape[i] != b_buf.shape[i]) {
            throw std::runtime_error("cuda add_forward: shape mismatch");
        }
    }

    ssize_t size_ssize = 1;

    for (ssize_t i = 0; i < a_buf.ndim; ++i) {
        size_ssize *= a_buf.shape[i];
    }

    int size = static_cast<int>(size_ssize);

    const double* a_ptr = static_cast<const double*>(a_buf.ptr);
    const double* b_ptr = static_cast<const double*>(b_buf.ptr);

    py::array_t<double> out(a_buf.shape);
    auto out_buf = out.request();
    double* out_ptr = static_cast<double*>(out_buf.ptr);

    size_t bytes = static_cast<size_t>(size) * sizeof(double);

    auto d_A = CudaArray::from_numpy(a);
    auto d_B = CudaArray::from_numpy(b);
    auto d_C = std::make_shared<CudaArray>(
        std::vector<ssize_t>{static_cast<ssize_t>(size)}
    );

    int block = 256;
    int grid = (size + block - 1) / block;

    add_kernel<<<grid, block>>>(d_A->ptr, d_B->ptr, d_C->ptr, size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        out_ptr,
        d_C->ptr,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    return out;
}


std::shared_ptr<CudaArray> add_storage(
    std::shared_ptr<CudaArray> a,
    std::shared_ptr<CudaArray> b
) {
    if (a->shape != b->shape) {
        throw std::runtime_error("add_storage: shape mismatch");
    }

    auto out = std::make_shared<CudaArray>(a->shape);

    int size = static_cast<int>(a->size);

    int block = 256;
    int grid = (size + block - 1) / block;

    add_kernel<<<grid, block>>>(a->ptr, b->ptr, out->ptr, size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return out;
}