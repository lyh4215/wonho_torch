from . import cpu

_current_backend = cpu

def set_backend(name):
    global _current_backend

    if name == "cpu":
        _current_backend = cpu
    elif name == "cuda":
        from . import cuda
        _current_backend = cuda
    else:
        raise ValueError(f"unknown backend: {name}")

def add(a, b):
    return _current_backend.add(a, b)

def matmul(a, b):
    return _current_backend.matmul(a, b)