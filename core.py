import numpy as np
from tensor import Tensor
class Parameter(Tensor):
    pass


class Module:
    def __init__(self):
        self.training = True
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def parameters(self):
        return []

    def state_dict(self, prefix=""):
        return {}
    
    def load_state_dict(self, state_dict, prefix=""):
        pass

    def train(self):
        self.training = True
    
    def eval(self):
        self.traning = False