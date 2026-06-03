from core import Module, Parameter
import numpy as np

class ReLU(Module):
    def __str__(self):
        return "ReLU()"

    def forward(self, x):
        self.input = x
        self.output = np.maximum(0, x)
        return self.output
    def backward(self, dy: np.ndarray):
        dx = dy * (self.input > 0)
        return dx

    def parameters(self):
        return []

class Sigmoid(Module):
    def __str__(self):
        return "Sigmoid()"

    def forward(self, x: np.ndarray):
        self.input = x
        self.output = 1/ (1 + np.exp(-x))
        return self.output
    
    def backward(self, dy : np.ndarray):
        dx = dy * self.output * (1- self.output)
        return dx
        


class Sequential(Module):
    def __init__(self, *layers : Module):
        self.layers : list[Module] = list(layers)

    def __repr__(self):
        output = ""
        for layer in self.layers:
            output += str(layer)
            output += "\n"
        return output

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, dy):
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def state_dict(self, prefix=""):
        state = {}

        for idx, layer in enumerate(self.layers):
            layer_prefix = prefix + f"{idx}."
            state.update(layer.state_dict(layer_prefix))

        return state

    def load_state_dict(self, state_dict, prefix=""):
        for idx, layer in enumerate(self.layers):
            layer_prefix = prefix + f"{idx}."
            layer.load_state_dict(state_dict, layer_prefix)
    

class Linear(Module):
    def __init__(self, input_dim: int, output_dim: int):
        self.lr = 10e-4
        self.W = Parameter(
            np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        )
        self.b = Parameter(np.zeros(output_dim,))
    def __repr__(self):
        return f"W: {self.W}\nB: {self.b}"

    def __str__(self):
        return f"W: {self.W}\nB: {self.b}"

    def forward(self, input : np.ndarray):
        self.input = input
        self.output = input @ self.W.data + self.b.data
        return self.output

    def backward(self, dy: np.ndarray):
        """
        dy = dL/dY
        """

        # input이 batch인 경우: input.shape == (batch_size, input_dim)
        if self.input.ndim == 2:
            self.W.grad = self.input.T @ dy
            self.b.grad = dy.sum(axis=0)
            dx = dy @ self.W.data.T

        # input이 단일 샘플인 경우: input.shape == (input_dim,)
        elif self.input.ndim == 1:
            self.W.grad = self.input[:, None] @ dy[None, :]
            self.b.grad = dy
            dx = dy @ self.W.data.T

        else:
            raise ValueError("input must be 1D or 2D numpy array")

        return dx

    def parameters(self):
        return [self.W, self.b]

    def state_dict(self, prefix=""):
        return {
            prefix + "W": self.W.data.copy(),
            prefix + "b": self.b.data.copy(),
        }
    
    def load_state_dict(self, state_dict, prefix=""):
        self.W.data = state_dict[prefix + "W"].copy()
        self.b.data = state_dict[prefix + "b"].copy()

        self.W.grad = np.zeros_like(self.W.data)
        self.b.grad = np.zeros_like(self.b.data)

def softmax(logits: np.ndarray):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

if __name__=="__main__":
    sigmoid = Sigmoid()

    x = np.array([-2.0, 0.0, 2.0])
    out = sigmoid.forward(x)

    print(out)
    # 대략 [0.119, 0.5, 0.881]

    dy = np.array([1.0, 1.0, 1.0])
    dx = sigmoid.backward(dy)

    print(dx)
    # 대략 [0.105, 0.25, 0.105]