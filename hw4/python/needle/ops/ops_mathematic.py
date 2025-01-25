"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


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
        if self.axes is not None:
            reduce_axes = []
            if isinstance(self.axes, int):
                self.axes = (self.axes,)
            reduce_axes = list(self.axes)
            for axis in reversed(sorted(reduce_axes)):
                a = array_api.sum(a, axis=(axis,))
            return a
        else:
            return array_api.sum(a)
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
            if isinstance(self.axes, int):
                self.axes = (self.axes,)
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
        return (out_grad / node.inputs[0],)
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
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out = node.realize_cached_data()
        return out_grad * Tensor(out > 0, device=out_grad.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # outgrad(1 - tanh^2)
        return out_grad - out_grad * (node * node)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: tuple) -> NDArray:
        ### BEGIN YOUR SOLUTION
        assert len(args) > 0, "Stack needs at least one item!"
        shape = args[0].shape
        for arg in args:
            assert shape == arg.shape, "All item needs to be of the same size!"

        length = len(args)
        newshape = list(shape)
        newshape.insert(self.axis, length)

        outArray = array_api.empty(shape=newshape, device=args[0].device)
        slices = [slice(0, shape) for shape in newshape]
        for i, arg in enumerate(args):
            slices[self.axis] = slice(i, i + 1)
            outArray[tuple(slices)] = arg
        return outArray
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (split(out_grad, self.axis),)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        shape = list(A.shape)
        length = shape[self.axis]
        slices = [slice(0, s) for s in shape]
        shape.pop(self.axis)

        outArray = []
        for i in range(length):
            slices[self.axis] = slice(i, i + 1)
            outArray.append(array_api.reshape(A[tuple(slices)], shape))
        return outArray
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (stack(out_grad, axis=self.axis),)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (flip(out_grad, self.axes),)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        newshape = list(a.shape)
        if isinstance(self.axes, int):
            self.axes = (self.axes,)

        for axis in self.axes:
            newshape[axis] = newshape[axis] * (self.dilation + 1)
        outArray = array_api.full(newshape, 0.0, device=a.device)
        slices = [slice(0, shape) for shape in newshape]
        for axis in self.axes:
            slices[axis] = slice(0, newshape[axis], self.dilation + 1)

        outArray[tuple(slices)] = a
        return outArray
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (undilate(out_grad, axes=self.axes, dilation=self.dilation),)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if isinstance(self.axes, int):
            self.axes = (self.axes,)

        slices = [slice(0, shape) for shape in a.shape]
        for axis in self.axes:
            slices[axis] = slice(0, a.shape[axis], self.dilation + 1)

        return a[tuple(slices)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (dilate(out_grad, axes=self.axes, dilation=self.dilation),)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        paddings = (
            (0, 0),
            (self.padding, self.padding),
            (self.padding, self.padding),
            (0, 0),
        )
        padded_A = A.pad(paddings)

        N, H, W, C_in0 = padded_A.shape
        K0, K1, C_in1, C_out = B.shape
        assert K0 == K1 and C_in0 == C_in1

        inner_dim = K0 * K0 * C_in0
        Ns, Hs, Ws, Cs = padded_A.strides

        # consider strides
        out_H = ((H - K0 + 1) + self.stride - 1) // self.stride
        out_W = ((W - K0 + 1) + self.stride - 1) // self.stride

        im2col = array_api.reshape(
            padded_A.as_strided(
                shape=(N, out_H, out_W, K0, K0, C_in0),
                strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs),
            ),
            ((N * out_H * out_W), inner_dim),
        )
        out = im2col @ array_api.reshape(B, (inner_dim, C_out))
        return array_api.reshape(out, (N, out_H, out_W, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        K, _, _, _ = B.shape
        assert (K - self.padding - 1) >= 0

        out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)

        # calc A.grad
        flip_B = flip(B, axes=(0, 1))
        flip_B = transpose(flip_B, axes=(2, 3))
        A_grad = conv(out_grad, flip_B, stride=1, padding=K - self.padding - 1)

        # calc B.grad
        out_grad = transpose(out_grad, axes=(0, 1))
        out_grad = transpose(out_grad, axes=(1, 2))

        B_grad = conv(
            transpose(A, axes=(0, 3)), out_grad, stride=1, padding=self.padding
        )  # C_in, k, k, C_out
        B_grad = transpose(B_grad, axes=(0, 1))
        B_grad = transpose(B_grad, axes=(1, 2))

        return (A_grad, B_grad)
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
