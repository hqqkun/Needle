"""The module.
"""

from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from functools import reduce


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, dtype=dtype, device=device)
        )
        self.bias = (
            Parameter(
                init.kaiming_uniform(
                    out_features, 1, dtype=dtype, device=device
                ).reshape((1, out_features))
            )
            if bias
            else None
        )

        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = ops.matmul(X, self.weight)
        if self.bias:
            out = out + self.bias.broadcast_to(out.shape)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor):
        ### BEGIN YOUR SOLUTION
        B = X.shape[0]
        numel = reduce(lambda x, y: x * y, X.shape, 1)
        return X.reshape((B, numel // B))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for module in self.modules:
            out = module(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        assert len(logits.shape) == 2 and len(y.shape) == 1
        assert logits.shape[0] == y.shape[0]

        batch, classes = logits.shape[0], logits.shape[1]
        sum_exp_Z = ops.logsumexp(logits, axes=(1,))
        y_one_hot = init.one_hot(classes, y)
        value_y = (logits * y_one_hot).sum(axes=(1,))
        sum_error = (sum_exp_Z - value_y).sum()
        return sum_error / batch
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim), dtype=dtype, device=device)
        self.bias = Parameter(init.zeros(dim), dtype=dtype, device=device)
        self.running_mean = init.zeros(dim, dtype=dtype, device=device)
        self.running_var = init.ones(dim, dtype=dtype, device=device)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        w1 = self.weight.broadcast_to(x.shape)
        b1 = self.bias.broadcast_to(x.shape)

        if self.training:
            batch = x.shape[0]
            mean = x.sum(axes=(0,)) / batch
            numerator = x - mean.broadcast_to(x.shape)
            VarX = (numerator * numerator).sum(axes=(0,)) / batch

            self.running_mean.data = (
                1 - self.momentum
            ) * self.running_mean.data + self.momentum * mean.data
            self.running_var.data = (
                1 - self.momentum
            ) * self.running_var.data + self.momentum * VarX.data
            denominator = (VarX.broadcast_to(x.shape) + self.eps) ** 0.5
            return (w1 * numerator) / denominator + b1

        else:
            mu = self.running_mean.broadcast_to(x.shape)
            std = (self.running_var + self.eps).broadcast_to(x.shape) ** 0.5
            out = (x - mu) / std
            return w1 * out + b1
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, dtype=dtype, device=device))
        self.bias = Parameter(init.zeros(dim, dtype=dtype, device=device))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch, layer = x.shape
        mean = (x.sum(axes=(1,)) / layer).reshape((batch, 1)).broadcast_to(x.shape)
        numerator = x - mean
        VarX = (
            ((numerator * numerator).sum(axes=(1,)) / layer)
            .reshape((batch, 1))
            .broadcast_to(x.shape)
        )
        denominator = (VarX + self.eps) ** 0.5
        w1 = self.weight.broadcast_to(x.shape)
        b1 = self.bias.broadcast_to(x.shape)

        return (w1 * numerator) / denominator + b1

        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1 - self.p)
            return (x * mask) / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
