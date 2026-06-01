import numpy as np
class Parameter:
    def __init__ (self, data : np.ndarray):
        self.data = data
        self.grad = np.zeros_like(data)
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def __repr__(self):
        return f"{self.data}"


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        pass
    
    def backward(self, dy):
        pass
    
    def parameters(self):
        return []

    def state_dict(self, prefix=""):
        return {}
    
    def load_state_dict(self, state_dict, prefix=""):
        pass