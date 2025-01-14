"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
from itertools import zip_longest
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node: Tensor):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_grad = out_grad * rhs * power(lhs, rhs - 1)
        rhs_grad = out_grad * node * log(lhs)
        return (lhs_grad, rhs_grad)
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        input_grad = (out_grad * (input ** (self.scalar - 1))) * self.scalar
        return (input_grad,)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        a_grad = out_grad / b
        b_grad = (-out_grad * a) / (b * b)
        return (a_grad, b_grad)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad / self.scalar,)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        if self.axes:
            assert len(self.axes) == 2
            return array_api.swapaxes(a, axis1=self.axes[0], axis2=self.axes[1])
        else:
            return array_api.swapaxes(a, axis1=-1, axis2=-2)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, newshape=self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return (reshape(out_grad, node.inputs[0].shape),)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, shape=self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        out_grad_shape = out_grad.shape
        input_ndim = len(input_shape)
        out_grad_ndim = len(out_grad_shape)
        assert out_grad_ndim >= input_ndim

        expand_input_shape = [1] * (out_grad_ndim - input_ndim) + list(input_shape)
        reduce_axes = []
        for i in range(out_grad_ndim - 1, -1, -1):
            if expand_input_shape[i] == out_grad_shape[i]:
                continue
            assert expand_input_shape[i] == 1
            reduce_axes.append(i)

        if len(reduce_axes) != 0:
            return (out_grad.sum(axes=tuple(reduce_axes)).reshape(input_shape),)
        else:
            return (out_grad,)

        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # find expand shape
        expand_shape = []
        orig_shape = node.inputs[0].shape
        orig_ndim = len(orig_shape)

        if self.axes is None:
            expand_shape = [1] * orig_ndim
        else:
            expand_shape = list(node.inputs[0].shape)
            for axis in self.axes:
                expand_shape[axis] = 1
        # then broadcast
        input_grad = out_grad.reshape(expand_shape).broadcast_to(orig_shape)
        return (input_grad,)

        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        ndim_a = len(a.shape)
        ndim_b = len(b.shape)

        a_grad = matmul(out_grad, transpose(b))
        b_grad = matmul(transpose(a), out_grad)

        if ndim_a != ndim_b:
            reduce_axes = tuple(range(abs(ndim_a - ndim_b)))
            if ndim_a > ndim_b:
                b_grad = b_grad.sum(axes=reduce_axes)
            else:
                a_grad = a_grad.sum(axes=reduce_axes)
        return (a_grad, b_grad)

        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-out_grad,)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad / node.inputs[0], )
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * node,)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a: NDArray):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(0, a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        mask = array_api.where(node.realize_cached_data() > 0, 1, 0)
        return (out_grad * mask,)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
