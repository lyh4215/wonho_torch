// wonho_torch/backend/native_ops.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

py::array_t<double> add_forward(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b
) {
    auto a_buf = a.request();
    auto b_buf = b.request();

    if (a_buf.ndim != b_buf.ndim) {
        throw std::runtime_error("add_forward: ndim mismatch");
    }

    for (ssize_t i = 0; i < a_buf.ndim; ++i) {
        if (a_buf.shape[i] != b_buf.shape[i]) {
            throw std::runtime_error("add_forward: shape mismatch");
        }
    }

    ssize_t size = 1;
    for (ssize_t i = 0; i < a_buf.ndim; ++i) {
        size *= a_buf.shape[i];
    }

    py::array_t<double> out(a_buf.shape);
    auto out_buf = out.request();

    double* a_ptr = static_cast<double*>(a_buf.ptr);
    double* b_ptr = static_cast<double*>(b_buf.ptr);
    double* out_ptr = static_cast<double*>(out_buf.ptr);

    for (ssize_t i = 0; i < size; ++i) {
        out_ptr[i] = a_ptr[i] + b_ptr[i];
    }

    return out;
}

py::array_t<double> matmul_forward(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b
) {
    auto a_buf = a.request();
    auto b_buf = b.request();

    if (a_buf.ndim != 2 || b_buf.ndim != 2) {
        throw std::runtime_error("matmul_forward: only 2D arrays are supported");
    }

    ssize_t M = a_buf.shape[0];
    ssize_t K = a_buf.shape[1];

    ssize_t K2 = b_buf.shape[0];
    ssize_t N = b_buf.shape[1];

    if (K != K2) {
        throw std::runtime_error("matmul_forward: shape mismatch");
    }

    py::array_t<double> out(std::vector<ssize_t>{M, N});
    auto out_buf = out.request();

    double* a_ptr = static_cast<double*>(a_buf.ptr);
    double* b_ptr = static_cast<double*>(b_buf.ptr);
    double* out_ptr = static_cast<double*>(out_buf.ptr);

    for (ssize_t i = 0; i < M; ++i) {
        for (ssize_t j = 0; j < N; ++j) {
            double sum = 0.0;

            for (ssize_t k = 0; k < K; ++k) {
                sum += a_ptr[i * K + k] * b_ptr[k * N + j];
            }

            out_ptr[i * N + j] = sum;
        }
    }

    return out;
}

PYBIND11_MODULE(_C, m) {
    m.def("add_forward", &add_forward, "Add two NumPy arrays using C++");
    m.def("matmul_forward", &matmul_forward, "Matrix multiplication using C++");
}