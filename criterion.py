import numpy as np
from core import Module
from tensor import Tensor

class MSELoss:
    def __call__(self, y_pred, y_true):
        y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)
        diff = y_pred - y_true
        return (diff * diff).mean()
    
class BCELoss:
    def __init__(self, eps=1e-7):
        self.eps = eps

    def __call__(self, y_pred, y_true):
        y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)

        loss = -(
            y_true * (y_pred + self.eps).log()
            + (1 - y_true) * (1 - y_pred + self.eps).log()
        ).mean()

        return loss

class SoftmaxCrossEntropyLoss:
    def __init__(self, eps=1e-12):
        self.eps = eps

    def __call__(self, logits, y_true):
        y_true = np.array(y_true, dtype=int)

        shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        batch_size = logits.data.shape[0]

        correct_probs = probs[np.arange(batch_size), y_true]
        loss_data = -np.mean(np.log(correct_probs + self.eps))

        out = Tensor(
            loss_data,
            _children=(logits,),
            _op="softmax_cross_entropy"
        )

        def _backward():
            dlogits = probs.copy()
            dlogits[np.arange(batch_size), y_true] -= 1
            dlogits /= batch_size

            logits.grad += dlogits * out.grad

        out._backward = _backward

        return out