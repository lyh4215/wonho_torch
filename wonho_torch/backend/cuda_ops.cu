// wonho_torch/backend/cuda_ops.cu

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>
#include <string>

namespace py = pybind11;

#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            throw std::runtime_error(                              \
                std::string("CUDA error: ") + cudaGetErrorString(err) \
            );                                                     \
        }                                                          \
    } while (0)

class CudaBuffer {
public:
    void* ptr;

    explicit CudaBuffer(size_t bytes) : ptr(nullptr) {
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
    }

    ~CudaBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    template <typename T>
    T* as() {
        return static_cast<T*>(ptr);
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
};

__global__ void matmul_kernel(
    const double* A,
    const double* B,
    double* C,
    int M,
    int K,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;

        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

py::array_t<double> matmul_forward(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b
) {
    auto a_buf = a.request();
    auto b_buf = b.request();

    if (a_buf.ndim != 2 || b_buf.ndim != 2) {
        throw std::runtime_error("cuda matmul_forward: only 2D arrays are supported");
    }

    int M = static_cast<int>(a_buf.shape[0]);
    int K = static_cast<int>(a_buf.shape[1]);

    int K2 = static_cast<int>(b_buf.shape[0]);
    int N = static_cast<int>(b_buf.shape[1]);

    if (K != K2) {
        throw std::runtime_error("cuda matmul_forward: shape mismatch");
    }

    const double* a_ptr = static_cast<const double*>(a_buf.ptr);
    const double* b_ptr = static_cast<const double*>(b_buf.ptr);

    py::array_t<double> out(std::vector<ssize_t>{M, N});
    auto out_buf = out.request();
    double* out_ptr = static_cast<double*>(out_buf.ptr);

    size_t a_bytes = static_cast<size_t>(M) * K * sizeof(double);
    size_t b_bytes = static_cast<size_t>(K) * N * sizeof(double);
    size_t c_bytes = static_cast<size_t>(M) * N * sizeof(double);

    CudaBuffer d_A(a_bytes);
    CudaBuffer d_B(b_bytes);
    CudaBuffer d_C(c_bytes);

    CUDA_CHECK(cudaMemcpy(
        d_A.as<double>(),
        a_ptr,
        a_bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B.as<double>(),
        b_ptr,
        b_bytes,
        cudaMemcpyHostToDevice
    ));

    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    matmul_kernel<<<grid, block>>>(
        d_A.as<double>(),
        d_B.as<double>(),
        d_C.as<double>(),
        M,
        K,
        N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        out_ptr,
        d_C.as<double>(),
        c_bytes,
        cudaMemcpyDeviceToHost
    ));

    return out;
}

PYBIND11_MODULE(_CUDA, m) {
    m.def("matmul_forward", &matmul_forward, "Naive CUDA matrix multiplication");
}