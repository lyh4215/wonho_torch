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
        throw std::runtime_error("cuda matmul_forward_naive: only 2D arrays are supported");
    }

    int M = static_cast<int>(a_buf.shape[0]);
    int K = static_cast<int>(a_buf.shape[1]);

    int K2 = static_cast<int>(b_buf.shape[0]);
    int N = static_cast<int>(b_buf.shape[1]);

    if (K != K2) {
        throw std::runtime_error("cuda matmul_forward_naive: shape mismatch");
    }

    auto d_A = CudaArray::from_numpy(a);
    auto d_B = CudaArray::from_numpy(b);
    auto d_C = std::make_shared<CudaArray>(
        std::vector<ssize_t>{
            static_cast<ssize_t>(M),
            static_cast<ssize_t>(N)
        }
    );

    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    matmul_naive_kernel<<<grid, block>>>(
        d_A->ptr,
        d_B->ptr,
        d_C->ptr,
        M,
        K,
        N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return d_C->to_numpy();
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

    auto d_A = CudaArray::from_numpy(a);
    auto d_B = CudaArray::from_numpy(b);
    auto d_C = std::make_shared<CudaArray>(
        std::vector<ssize_t>{
            static_cast<ssize_t>(M),
            static_cast<ssize_t>(N)
        }
    );

    dim3 block(TILE, TILE);
    dim3 grid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    matmul_tiled_kernel<<<grid, block>>>(
        d_A->ptr,
        d_B->ptr,
        d_C->ptr,
        M,
        K,
        N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return d_C->to_numpy();
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

    auto d_A = CudaArray::from_numpy(a);
    auto d_B = CudaArray::from_numpy(b);
    auto d_C = std::make_shared<CudaArray>(
        std::vector<ssize_t>{
            static_cast<ssize_t>(M),
            static_cast<ssize_t>(N)
        }
    );

    static CublasHandle blas;

    double alpha = 1.0;
    double beta = 0.0;

    CUBLAS_CHECK(cublasDgemm(
        blas.handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        d_B->ptr,
        N,
        d_A->ptr,
        K,
        &beta,
        d_C->ptr,
        N
    ));

    CUDA_CHECK(cudaDeviceSynchronize());

    return d_C->to_numpy();
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