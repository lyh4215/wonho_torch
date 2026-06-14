import numpy as np


class CPUStorage:
    def __init__(self, data):
        if isinstance(data, CPUStorage):
            self.array = data.array
        else:
            self.array = np.asarray(data, dtype=np.float64)

    @property
    def device(self):
        return "cpu"

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    def numpy(self):
        return self.array

    def __repr__(self):
        return f"CPUStorage(shape={self.shape}, dtype=float64)"


class CUDAStorage:
    def __init__(self, data):
        from wonho_torch.backend import cuda

        if isinstance(data, CUDAStorage):
            self.array = data.array
        elif hasattr(data, "to_numpy"):
            # _CUDA.CudaArray
            self.array = data
        else:
            # CPU data -> GPU CudaArray
            self.array = cuda.array_from_numpy(
                np.asarray(data, dtype=np.float64)
            )

    @classmethod
    def from_cuda_array(cls, cuda_array):
        obj = cls.__new__(cls)
        obj.array = cuda_array
        return obj

    @property
    def device(self):
        return "cuda"

    @property
    def shape(self):
        return tuple(self.array.shape)

    @property
    def ndim(self):
        return len(self.shape)

    def numpy(self):
        return self.array.to_numpy()

    def __repr__(self):
        return f"CUDAStorage(shape={self.shape}, dtype=float64)"