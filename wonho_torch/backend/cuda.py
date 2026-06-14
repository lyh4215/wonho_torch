import numpy as np

try:
    from wonho_torch.backend import _CUDA
except ImportError as e:
    _CUDA = None
    _CUDA_IMPORT_ERROR = e


def _require_cuda():
    if _CUDA is None:
        raise RuntimeError(
            "CUDA backend is not available. "
            "Build wonho_torch on an environment with CUDA Toolkit/nvcc."
        ) from _CUDA_IMPORT_ERROR


# =========================
# CudaArray / Storage helpers
# =========================

def array_from_numpy(a):
    """
    CPU np.ndarray -> _CUDA.CudaArray
    여기서 H2D copy 발생.
    """
    _require_cuda()

    return _CUDA.CudaArray.from_numpy(
        np.ascontiguousarray(a, dtype=np.float64)
    )


def to_numpy(cuda_array):
    """
    _CUDA.CudaArray -> CPU np.ndarray
    여기서 D2H copy 발생.
    """
    _require_cuda()

    return cuda_array.to_numpy()


def add_storage(a_storage, b_storage):
    """
    CUDAStorage + CUDAStorage -> _CUDA.CudaArray

    a_storage.array, b_storage.array는 _CUDA.CudaArray라고 가정.
    결과도 GPU에 남아있는 CudaArray.
    """
    _require_cuda()

    return _CUDA.add_storage(
        a_storage.array,
        b_storage.array,
    )


def matmul_storage(a_storage, b_storage):
    """
    CUDAStorage @ CUDAStorage -> _CUDA.CudaArray

    여기서는 H2D/D2H copy 없음.
    이미 GPU에 있는 포인터끼리 cuBLAS 실행.
    """
    _require_cuda()

    return _CUDA.matmul_storage_cublas(
        a_storage.array,
        b_storage.array,
    )


# =========================
# Old roundtrip functions
# np.ndarray -> GPU -> np.ndarray
# 비교/벤치마크용으로 유지
# =========================

def add(a, b):
    """
    기존 Tensor backend.add 용.
    CPU ndarray 입력 -> GPU 계산 -> CPU ndarray 반환.
    broadcasting은 Python/NumPy에서 처리.
    """
    _require_cuda()

    a_broad, b_broad = np.broadcast_arrays(a, b)

    return _CUDA.add_forward(
        np.ascontiguousarray(a_broad, dtype=np.float64),
        np.ascontiguousarray(b_broad, dtype=np.float64),
    )


def matmul_naive(a, b):
    _require_cuda()

    return _CUDA.matmul_forward(
        np.ascontiguousarray(a, dtype=np.float64),
        np.ascontiguousarray(b, dtype=np.float64),
    )


def matmul_tiled(a, b):
    _require_cuda()

    return _CUDA.matmul_forward_tiled(
        np.ascontiguousarray(a, dtype=np.float64),
        np.ascontiguousarray(b, dtype=np.float64),
    )


def matmul_cublas(a, b):
    _require_cuda()

    return _CUDA.matmul_forward_cublas(
        np.ascontiguousarray(a, dtype=np.float64),
        np.ascontiguousarray(b, dtype=np.float64),
    )


def matmul(a, b):
    """
    기존 backend.matmul 용.
    아직 Tensor.data=np.ndarray 기반 코드가 이걸 씀.
    기본은 cuBLAS roundtrip으로 둔다.
    """
    return matmul_cublas(a, b)