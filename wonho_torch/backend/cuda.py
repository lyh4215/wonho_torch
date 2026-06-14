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


def add(a, b):
    _require_cuda()

    a_broad, b_broad = np.broadcast_arrays(a, b)

    return _CUDA.add_forward(
        np.ascontiguousarray(a_broad),
        np.ascontiguousarray(b_broad),
    )


def matmul(a, b):
    _require_cuda()

    return _CUDA.matmul_forward(
        np.ascontiguousarray(a),
        np.ascontiguousarray(b),
    )

def matmul_tiled(a, b):
    _require_cuda()

    return _CUDA.matmul_forward_tiled(
        np.ascontiguousarray(a),
        np.ascontiguousarray(b),
    )