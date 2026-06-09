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
        super().__init__()
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

    def train(self):
        self.training= True
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

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask=None

    def forward(self, x):
        if not self.training:
            return x

        # keep probability
        keep_prob = 1.0 - self.p

        self.mask = (
            np.random.rand(*x.shape) < keep_prob
        ) / keep_prob

        return x * self.mask

    def backward(self, dy):
        if not self.training:
            return dy

        return dy * self.mask

class BatchNorm1D(Module):
    def __init__(self, dim, momentum=0.9, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.momentum = momentum
        self.eps = eps

        self.gamma = Parameter(np.ones(dim))
        self.beta = Parameter(np.zeros(dim))

        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)

    def forward(self, x):
        self.x = x
        if self.training:
            self.batch_mean = x.mean(axis=0)
            self.batch_var = x.var(axis=0)

            self.x_centered = x - self.batch_mean
            self.std_inv = 1.0 / np.sqrt(self.batch_var + self.eps)
            self.x_hat = self.x_centered * self.std_inv

            out = self.gamma.data * self.x_hat + self.beta.data

            self.running_mean = (
                self.momentum * self.running_mean
                + (1 - self.momentum) * self.batch_mean
            )

            self.running_var = (
                self.momentum * self.running_var
                + (1 - self.momentum) * self.batch_var
            )

            return out

        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma.data * x_hat + self.beta.data
            return out

    def backward(self, dout):
        """
        dout: dL/dout, shape (N, D)
        x:    shape (N, D)
        """

        N = dout.shape[0]

        self.gamma.grad = np.sum(dout * self.x_hat, axis=0)
        self.beta.grad = np.sum(dout, axis=0)

        dx_hat = dout * self.gamma.data

        dx = (1.0 / N) * self.std_inv * (
            N * dx_hat
            - np.sum(dx_hat, axis=0)
            - self.x_hat * np.sum(dx_hat * self.x_hat, axis=0)
        )

        return dx

    def parameters(self):
        return [self.gamma, self.beta]

    def __str__(self):
        return f"BatchNorm1D(dim={self.dim})"

def softmax(logits: np.ndarray):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)



class Conv2D(Module):
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
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        self.x = x

        N, C, H, W = x.shape
        K = self.kernel_size
        S = self.stride

        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1

        out = np.zeros((N, C, H_out, W_out))
        self.max_indices = {}

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        h_end = h_start + K
                        w_start = j * S
                        w_end = w_start + K

                        region = x[n, c, h_start:h_end, w_start:w_end]
                        out[n, c, i, j] = np.max(region)

                        max_idx = np.unravel_index(np.argmax(region), region.shape)
                        self.max_indices[(n, c, i, j)] = (
                            h_start + max_idx[0],
                            w_start + max_idx[1]
                        )

        return out

    def backward(self, dout):
        x = self.x
        N, C, H, W = x.shape

        dx = np.zeros_like(x)

        _, _, H_out, W_out = dout.shape

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_idx, w_idx = self.max_indices[(n, c, i, j)]
                        dx[n, c, h_idx, w_idx] += dout[n, c, i, j]

        return dx

    
class Flatten(Module):
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.input_shape)

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