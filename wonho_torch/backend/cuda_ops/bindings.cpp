#include <pybind11/pybind11.h>

#include "cuda_array.h"
#include "add.h"
#include "matmul.h"

namespace py = pybind11;

PYBIND11_MODULE(_CUDA, m) {
    py::class_<CudaArray, std::shared_ptr<CudaArray>>(m, "CudaArray")
        .def_static("from_numpy", &CudaArray::from_numpy)
        .def("to_numpy", &CudaArray::to_numpy)
        .def_property_readonly("shape", [](const CudaArray& arr) {
            return arr.shape;
        })
        .def("__repr__", &CudaArray::repr);

    m.def("add_forward", &add_forward, "Naive CUDA add with CPU roundtrip");
    m.def("add_storage", &add_storage, "CUDA-resident add");

    m.def("matmul_forward", &matmul_forward_naive, "Naive CUDA matmul with CPU roundtrip");
    m.def("matmul_forward_tiled", &matmul_forward_tiled, "Tiled CUDA matmul with CPU roundtrip");
    m.def("matmul_forward_cublas", &matmul_forward_cublas, "cuBLAS matmul with CPU roundtrip");

    m.def("matmul_storage_cublas", &matmul_storage_cublas, "CUDA-resident cuBLAS matmul");
}