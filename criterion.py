import numpy as np
from core import Module

class MSELoss(Module):
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        self.y_pred = y_pred
        self.y_true = y_true

        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    def backward(self):
        dy = 2 * (self.y_pred - self.y_true) / self.y_pred.size
        return dy