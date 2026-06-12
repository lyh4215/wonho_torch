# backend/cpu.py

from wonho_torch.backend import _C
import numpy as np

def add(a, b):
    a_broad, b_broad = np.broadcast_arrays(a, b)

    return _C.add_forward(
        np.ascontiguousarray(a_broad),
        np.ascontiguousarray(b_broad),
    )


def matmul(a, b):
    return _C.matmul_forward(
        np.ascontiguousarray(a),
        np.ascontiguousarray(b),
    )