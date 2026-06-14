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

__global__ void add_kernel(
    const double* A,
    const double* B,
    double* C,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

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

    CudaBuffer d_A(bytes);
    CudaBuffer d_B(bytes);
    CudaBuffer d_C(bytes);

    CUDA_CHECK(cudaMemcpy(
        d_A.as<double>(),
        a_ptr,
        bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B.as<double>(),
        b_ptr,
        bytes,
        cudaMemcpyHostToDevice
    ));

    int block = 256;
    int grid = (size + block - 1) / block;

    add_kernel<<<grid, block>>>(
        d_A.as<double>(),
        d_B.as<double>(),
        d_C.as<double>(),
        size
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        out_ptr,
        d_C.as<double>(),
        bytes,
        cudaMemcpyDeviceToHost
    ));

    return out;
}

#define TILE 16

__global__ void matmul_tiled_kernel(
    const double* A,
    const double* B,
    double* C,
    int M,
    int K,
    int N
) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    double sum = 0.0;

    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < TILE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

py::array_t<double> matmul_forward_tiled(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b
) {
    auto a_buf = a.request();
    auto b_buf = b.request();

    if (a_buf.ndim != 2 || b_buf.ndim != 2) {
        throw std::runtime_error("cuda matmul_forward_tiled: only 2D arrays are supported");
    }

    int M = static_cast<int>(a_buf.shape[0]);
    int K = static_cast<int>(a_buf.shape[1]);

    int K2 = static_cast<int>(b_buf.shape[0]);
    int N = static_cast<int>(b_buf.shape[1]);

    if (K != K2) {
        throw std::runtime_error("cuda matmul_forward_tiled: shape mismatch");
    }

    const double* a_ptr = static_cast<const double*>(a_buf.ptr);
    const double* b_ptr = static_cast<const double*>(b_buf.ptr);

    py::array_t<double> out(std::vector<ssize_t>{
        static_cast<ssize_t>(M),
        static_cast<ssize_t>(N)
    });

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

    dim3 block(TILE, TILE);
    dim3 grid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    matmul_tiled_kernel<<<grid, block>>>(
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
    m.def("add_forward", &add_forward, "Naive CUDA add");
    m.def("matmul_forward", &matmul_forward, "Naive CUDA matrix multiplication");
    m.def("matmul_forward_tiled", &matmul_forward_tiled, "Shared memory tiled CUDA matrix multiplication");
}