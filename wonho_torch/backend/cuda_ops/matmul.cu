#include "matmul.h"
#include "common.cuh"
#include "kernels.cuh"

#include <stdexcept>
#include <vector>


py::array_t<double> matmul_forward_naive(
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

py::array_t<double> matmul_forward_cublas(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b
) {
    auto a_buf = a.request();
    auto b_buf = b.request();

    if (a_buf.ndim != 2 || b_buf.ndim != 2) {
        throw std::runtime_error("cuda matmul_forward_cublas: only 2D arrays are supported");
    }

    int M = static_cast<int>(a_buf.shape[0]);
    int K = static_cast<int>(a_buf.shape[1]);

    int K2 = static_cast<int>(b_buf.shape[0]);
    int N = static_cast<int>(b_buf.shape[1]);

    if (K != K2) {
        throw std::runtime_error("cuda matmul_forward_cublas: shape mismatch");
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

    static CublasHandle blas;

    double alpha = 1.0;
    double beta = 0.0;

    /*
        We want row-major:
            C[M, N] = A[M, K] @ B[K, N]

        cuBLAS assumes column-major.

        Row-major C[M, N] has the same memory layout as
        column-major C_col[N, M], which represents C^T.

        So we compute:
            C^T = B^T @ A^T

        cuBLAS sees:
            B as column-major matrix of shape (N, K)
            A as column-major matrix of shape (K, M)
            C as column-major matrix of shape (N, M)

        Therefore:
            m = N
            n = M
            k = K
            lda = N
            ldb = K
            ldc = N
    */

    CUBLAS_CHECK(cublasDgemm(
        blas.handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,          // m
        M,          // n
        K,          // k
        &alpha,
        d_B.as<double>(),
        N,          // lda
        d_A.as<double>(),
        K,          // ldb
        &beta,
        d_C.as<double>(),
        N           // ldc
    ));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        out_ptr,
        d_C.as<double>(),
        c_bytes,
        cudaMemcpyDeviceToHost
    ));

    return out;
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


std::shared_ptr<CudaArray> matmul_storage_cublas(
    std::shared_ptr<CudaArray> a,
    std::shared_ptr<CudaArray> b
) {
    if (a->shape.size() != 2 || b->shape.size() != 2) {
        throw std::runtime_error("matmul_storage_cublas: only 2D arrays are supported");
    }

    int M = static_cast<int>(a->shape[0]);
    int K = static_cast<int>(a->shape[1]);

    int K2 = static_cast<int>(b->shape[0]);
    int N = static_cast<int>(b->shape[1]);

    if (K != K2) {
        throw std::runtime_error("matmul_storage_cublas: shape mismatch");
    }

    auto out = std::make_shared<CudaArray>(
        std::vector<ssize_t>{
            static_cast<ssize_t>(M),
            static_cast<ssize_t>(N)
        }
    );

    static CublasHandle blas;

    double alpha = 1.0;
    double beta = 0.0;

    /*
        Row-major:
            C[M, N] = A[M, K] @ B[K, N]

        cuBLAS column-major trick:
            C^T[N, M] = B^T[N, K] @ A^T[K, M]

        So cuBLAS sees:
            B as column-major (N, K)
            A as column-major (K, M)
            C as column-major (N, M)
    */
    CUBLAS_CHECK(cublasDgemm(
        blas.handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,          // m
        M,          // n
        K,          // k
        &alpha,
        b->ptr,
        N,          // lda
        a->ptr,
        K,          // ldb
        &beta,
        out->ptr,
        N           // ldc
    ));

    CUDA_CHECK(cudaDeviceSynchronize());

    return out;
}