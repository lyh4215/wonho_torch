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
    
class BCELoss(Module): #BinaryCrossEntropyLoss
    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self.eps = 1e-7

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        self.y_pred = np.clip(y_pred, self.eps, 1- self.eps)
        self.y_true = y_true

        loss = -np.mean(
            y_true * np.log(self.y_pred)
            + (1 - y_true) * np.log(1- self.y_pred)
        )

        return loss
    
    def backward(self):

        dy = (
            -(self.y_true / self.y_pred)
            + ((1-self.y_true) / (1- self.y_pred))
        ) / self.y_pred.size

        return dy