import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data + other.data,
            _children=(self, other),
            _op="+"
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

    def sum(self):
        out = Tensor(
            self.data.sum(),
            _children=(self,),
            _op="sum"
        )

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def mean(self):
        out = Tensor(
            self.data.mean(),
            _children=(self,),
            _op="mean"
        )

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad / self.data.size

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data @ other.data,
            _children=(self, other),
            _op="@"
        )

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out

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

    def transpose(self):
        out = Tensor(
            self.data.T,
            _children=(self,),
            _op="transpose"
        )

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

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

    y = x.relu()
    z = (y * y).sum()
    z.backward()

    print(z.data)   # 10.0
    print(x.grad)   # [2. 0. 6.]