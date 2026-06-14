#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            throw std::runtime_error(                              \
                std::string("CUDA error: ") + cudaGetErrorString(err) \
            );                                                     \
        }                                                          \
    } while (0)

#define CUBLAS_CHECK(call)                                         \
    do {                                                           \
        cublasStatus_t status = call;                              \
        if (status != CUBLAS_STATUS_SUCCESS) {                     \
            throw std::runtime_error(                              \
                std::string("cuBLAS error: ") +                   \
                std::to_string(static_cast<int>(status))           \
            );                                                     \
        }                                                          \
    } while (0)


class CublasHandle {
public:
    cublasHandle_t handle;

    CublasHandle() : handle(nullptr) {
        CUBLAS_CHECK(cublasCreate(&handle));
    }

    ~CublasHandle() {
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
};