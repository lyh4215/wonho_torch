import numpy as np
from wonho_torch import backend

class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out_data = backend.add(self.data, other.data)

        out = Tensor(
            out_data,
            _children=(self, other),
            _op="add_cpp"
        )

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data * other.data,
            _children=(self, other),
            _op="*"
        )

        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __pow__(self, power):
        assert isinstance(power, (int, float))

        out = Tensor(
            self.data ** power,
            _children=(self,),
            _op=f"**{power}"
        )

        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other
    
    def relu(self):
        out = Tensor(
            np.maximum(0, self.data),
            _children=(self,),
            _op="ReLU"
        )

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            _children=(self,),
            _op="sum"
        )

        def _backward():
            grad = out.grad

            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)

            self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out


    def mean(self, axis=None, keepdims=False):
        denom = self.data.size if axis is None else self.data.shape[axis]

        return self.sum(axis=axis, keepdims=keepdims) / denom

    def sqrt(self):
        return self ** 0.5

    def var(self, axis=None, keepdims=False):
        mean = self.mean(axis=axis, keepdims=True)
        diff = self - mean
        var = (diff * diff).mean(axis=axis, keepdims=keepdims)
        return var

    def max(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.max(axis=axis, keepdims=keepdims),
            _children=(self,),
            _op="max"
        )

        def _backward():
            max_data = self.data.max(axis=axis, keepdims=True)

            mask = self.data == max_data

            grad = out.grad

            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)

            # max가 여러 개면 gradient를 나눠줌
            count = mask.sum(axis=axis, keepdims=True)

            self.grad += mask * grad / count

        out._backward = _backward
        return out

    def std(self, axis=None, keepdims=False, eps=1e-7):
        return (self.var(axis=axis, keepdims=keepdims) + eps).sqrt()

    def __len__(self):
        return len(self.data)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out_data = backend.matmul(self.data, other.data)

        out = Tensor(
            out_data,
            _children=(self, other),
            _op="matmul_cpp"
        )

        def _backward():
            if self.requires_grad:
                self.grad += backend.matmul(
                    out.grad,
                    other.data.T
                )

            if other.requires_grad:
                other.grad += backend.matmul(
                    self.data.T,
                    out.grad
                )

        out._backward = _backward
        return out


    def __rmatmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other @ self

    def clip(self, min_value, max_value):
        out = Tensor(
            np.clip(self.data, min_value, max_value),
            _children=(self,),
            _op="clip"
        )

        def _backward():
            mask = (self.data >= min_value) & (self.data <= max_value)
            self.grad += mask * out.grad

        out._backward = _backward
        return out
    
    def __getitem__(self, idx):
        out = Tensor(
            self.data[idx],
            _children=(self,),
            _op="getitem"
        )

        def _backward():
            np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out
    
    def item(self):
        return self.data.item()

    def reshape(self, *shape):
        out = Tensor(
            self.data.reshape(*shape),
            _children=(self,),
            _op="reshape"
        )

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = tuple(reversed(range(self.data.ndim)))

        out = Tensor(
            np.transpose(self.data, axes),
            _children=(self,),
            _op="transpose"
        )

        inv_axes = np.argsort(axes)

        def _backward():
            self.grad += np.transpose(out.grad, inv_axes)

        out._backward = _backward
        return out


    def log(self):
        out = Tensor(
            np.log(self.data),
            _children=(self,),
            _op="log"
        )

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(
            np.exp(self.data),
            _children=(self,),
            _op="exp"
        )

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def softmax(self, axis=1):
        # max는 stability용. 여기서는 gradient 추적 안 해도 실용상 OK.
        shifted = self - Tensor(np.max(self.data, axis=axis, keepdims=True))
        exp_x = shifted.exp()
        return exp_x / exp_x.sum(axis=axis, keepdims=True)

    @property
    def T(self):
        return self.transpose()

    @property
    def shape(self):
        return self.data.shape

    def backward(self):
        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)

                for child in t._prev:
                    build_topo(child)

                topo.append(t)

        build_topo(self)

        self.grad = 1.0

        for t in reversed(topo):
            t._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def argmax(self, axis=None):
        return np.argmax(self.data, axis=axis)


def unbroadcast(grad, shape):
    # grad를 원래 shape로 줄인다

    # 1. grad 차원이 더 많으면 앞쪽 축을 sum
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    # 2. 원래 shape에서 1이었던 축은 broadcast된 축이므로 sum
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad

if __name__ == "__main__":
    x = Tensor([1.0, -2.0, 3.0])

    y = Tensor([1.0, 2.1, 3])
    z = x + y
    print(z)

    A = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])

    B = np.array([[10.0, 20.0],
                [30.0, 40.0],
                [50.0, 60.0]])

    C = backend.matmul(A, B)

    print(C)
    print(A @ B)