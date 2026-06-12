from wonho_torch.core import Module, Parameter
import numpy as np
from wonho_torch.tensor import Tensor

class ReLU(Module):
    def forward(self, x):
        return x.relu()

    def __str__(self):
        return "ReLU()"

class Sigmoid(Module):
    def forward(self, x):
        return 1 / (1 + (-x).exp())

    def __str__(self):
        return "Sigmoid()"
        


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

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

    def train(self):
        self.training = True
        for layer in self.layers:
            layer.train()

    def eval(self):
        self.training = False
        for layer in self.layers:
            layer.eval()

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
        return input @ self.W + self.b

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

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x

        keep_prob = 1.0 - self.p

        mask = (
            np.random.rand(*x.data.shape) < keep_prob
        ).astype(float) / keep_prob

        return x * Tensor(mask)

    def __str__(self):
        return f"Dropout(p={self.p})"

class BatchNorm1D(Module):
    def __init__(self, dim, momentum=0.9, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.momentum = momentum
        self.eps = eps

        self.gamma = Parameter(np.ones((1, dim)))
        self.beta = Parameter(np.zeros((1, dim)))

        self.running_mean = np.zeros((1, dim))
        self.running_var = np.ones((1, dim))

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(axis=0, keepdims=True)

            x_centered = x - batch_mean
            batch_var = (x_centered * x_centered).mean(axis=0, keepdims=True)

            x_hat = x_centered / ((batch_var + self.eps) ** 0.5)

            out = self.gamma * x_hat + self.beta

            # running stats는 autograd 추적 필요 없음
            self.running_mean = (
                self.momentum * self.running_mean
                + (1 - self.momentum) * batch_mean.data
            )

            self.running_var = (
                self.momentum * self.running_var
                + (1 - self.momentum) * batch_var.data
            )

            return out

        else:
            x_hat = (x - Tensor(self.running_mean)) / (
                (Tensor(self.running_var) + self.eps) ** 0.5
            )

            return self.gamma * x_hat + self.beta

    def parameters(self):
        return [self.gamma, self.beta]

    def state_dict(self, prefix=""):
        return {
            prefix + "gamma": self.gamma.data.copy(),
            prefix + "beta": self.beta.data.copy(),
            prefix + "running_mean": self.running_mean.copy(),
            prefix + "running_var": self.running_var.copy(),
        }

    def load_state_dict(self, state_dict, prefix=""):
        self.gamma.data = state_dict[prefix + "gamma"].copy()
        self.beta.data = state_dict[prefix + "beta"].copy()
        self.running_mean = state_dict[prefix + "running_mean"].copy()
        self.running_var = state_dict[prefix + "running_var"].copy()

        self.gamma.zero_grad()
        self.beta.zero_grad()

    def __str__(self):
        return f"BatchNorm1D(dim={self.dim})"


def softmax(logits: np.ndarray):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

from wonho_torch.im2col import *
class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        KH = KW = kernel_size

        self.W = Parameter(
            np.random.randn(out_channels, in_channels, KH, KW)
            * np.sqrt(2.0 / (in_channels * KH * KW))
        )
        self.b = Parameter(np.zeros(out_channels))

    def forward(self, x):
        N, C, H, W = x.data.shape
        F, _, KH, KW = self.W.data.shape

        H_out = (H + 2 * self.padding - KH) // self.stride + 1
        W_out = (W + 2 * self.padding - KW) // self.stride + 1

        X_col = im2col_tensor(
            x,
            KH,
            KW,
            stride=self.stride,
            padding=self.padding
        )

        W_col = self.W.reshape(F, -1).transpose(1, 0)

        out = X_col @ W_col + self.b

        out = out.reshape(N, H_out, W_out, F)
        out = out.transpose(0, 3, 1, 2)

        return out

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
        self.W.zero_grad()
        self.b.zero_grad()

    def __str__(self):
        return (
            f"Conv2D("
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding})"
        )

class OldConv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        KH = KW = kernel_size

        self.W = Parameter(
            np.random.randn(out_channels, in_channels, KH, KW)
            * np.sqrt(2.0 / (in_channels * KH * KW))
        )
        self.b = Parameter(np.zeros(out_channels))

    def forward(self, x):
        self.x = x

        N, C, H, W = x.shape
        F, _, KH, KW = self.W.data.shape

        P = self.padding
        S = self.stride

        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (P, P), (P, P)),
            mode="constant"
        )
        self.x_padded = x_padded

        H_out = (H + 2 * P - KH) // S + 1
        W_out = (W + 2 * P - KW) // S + 1

        out = np.zeros((N, F, H_out, W_out))

        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        h_end = h_start + KH
                        w_start = j * S
                        w_end = w_start + KW

                        region = x_padded[n, :, h_start:h_end, w_start:w_end]

                        out[n, f, i, j] = (
                            np.sum(region * self.W.data[f])
                            + self.b.data[f]
                        )

        return out

    def backward(self, dout):
        x = self.x
        x_padded = self.x_padded

        N, C, H, W = x.shape
        F, _, KH, KW = self.W.data.shape

        P = self.padding
        S = self.stride

        _, _, H_out, W_out = dout.shape

        dx_padded = np.zeros_like(x_padded)
        self.W.grad = np.zeros_like(self.W.data)
        self.b.grad = np.zeros_like(self.b.data)

        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        h_end = h_start + KH
                        w_start = j * S
                        w_end = w_start + KW

                        region = x_padded[n, :, h_start:h_end, w_start:w_end]

                        self.W.grad[f] += region * dout[n, f, i, j]
                        self.b.grad[f] += dout[n, f, i, j]

                        dx_padded[n, :, h_start:h_end, w_start:w_end] += (
                            self.W.data[f] * dout[n, f, i, j]
                        )

        if P > 0:
            dx = dx_padded[:, :, P:-P, P:-P]
        else:
            dx = dx_padded

        return dx
    def parameters(self):
        return [self.W, self.b]

class MaxPool2D(Module):
    def __init__(self, kernel_size=2, stride=None):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def maxpool2d_tensor(self, x, kernel_size=2, stride=None):
        if stride is None:
            stride = kernel_size

        N, C, H, W = x.data.shape
        K = kernel_size
        S = stride

        assert K == S, "현재 버전은 kernel_size == stride만 지원"
        assert H % K == 0 and W % K == 0

        x_reshaped = x.data.reshape(N, C, H // K, K, W // K, K)

        out_data = x_reshaped.max(axis=(3, 5))

        out = Tensor(
            out_data,
            _children=(x,),
            _op="maxpool2d"
        )

        max_mask = x_reshaped == out_data[:, :, :, None, :, None]

        def _backward():
            dx_reshaped = max_mask * out.grad[:, :, :, None, :, None]
            dx = dx_reshaped.reshape(x.data.shape)

            x.grad += dx

        out._backward = _backward

        return out
    def forward(self, x):
        return self.maxpool2d_tensor(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride
        )

    def __str__(self):
        return f"MaxPool2D(kernel_size={self.kernel_size}, stride={self.stride})"

class FastMaxPool2D(Module):
    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        self.x = x

        N, C, H, W = x.shape
        K = self.kernel_size
        S = self.stride

        assert K == S
        assert H % K == 0
        assert W % K == 0

        # (N, C, H, W)
        # -> (N, C, H//K, K, W//K, K)
        self.x_reshaped = x.reshape(N, C, H // K, K, W // K, K)

        out = self.x_reshaped.max(axis=(3, 5))

        self.max_mask = self.x_reshaped == out[:, :, :, None, :, None]

        return out

    def backward(self, dout):
        K = self.kernel_size

        # dout: (N, C, H//K, W//K)
        dx_reshaped = self.max_mask * dout[:, :, :, None, :, None]

        return dx_reshaped.reshape(self.x.shape)

class Flatten(Module):
    def forward(self, x):
        return x.reshape(x.data.shape[0], -1)

    def __str__(self):
        return "Flatten()"
    
class SimpleCNN(Module):
    def __init__(self):
        self.features = Sequential(
            Conv2D(1, 4, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(2, stride=2),

            Conv2D(4, 8, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(2, stride=2),
        )

        self.classifier = Sequential(
            Flatten(),
            Linear(8 * 7 * 7, 10),
            # BatchNorm1D(128),
            # ReLU(),
            # Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def backward(self, dy):
        dy = self.classifier.backward(dy)
        dy = self.features.backward(dy)
        return dy

    def parameters(self):
        params = []

        params.extend(self.features.parameters())
        params.extend(self.classifier.parameters())

        return params

    def train(self):
        self.features.train()
        self.classifier.train()

    def eval(self):
        self.features.eval()
        self.classifier.eval()

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