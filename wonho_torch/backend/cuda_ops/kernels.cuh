#pragma once

#define TILE 16

__global__ void add_kernel(
    const double* A,
    const double* B,
    double* C,
    int size
);

__global__ void matmul_naive_kernel(
    const double* A,
    const double* B,
    double* C,
    int M,
    int K,
    int N
);

__global__ void matmul_tiled_kernel(
    const double* A,
    const double* B,
    double* C,
    int M,
    int K,
    int N
);