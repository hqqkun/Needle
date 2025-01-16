from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND


class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray):
        ### BEGIN YOUR SOLUTION
        maxZ_keep = array_api.max(Z, axis=(1,), keepdims=True)
        sumExp = array_api.sum(array_api.exp(Z - maxZ_keep), axis=(1,), keepdims=True)
        _logsumexp = array_api.log(sumExp) + maxZ_keep
        return Z - _logsumexp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad - out_grad * exp(node)
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z: NDArray):
        ### BEGIN YOUR SOLUTION
        maxZ_keep = array_api.max(Z, axis=self.axes, keepdims=True).broadcast_to(
            Z.shape
        )
        maxZ = array_api.max(Z, axis=self.axes, keepdims=False)
        sumExp = array_api.sum(array_api.exp(Z - maxZ_keep), axis=self.axes)
        return array_api.log(sumExp) + maxZ
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        expand_shape = []
        input = node.inputs[0]
        orig_shape = input.shape
        orig_ndim = len(orig_shape)

        # find expand shape
        if self.axes is None:
            expand_shape = [1] * orig_ndim
        else:
            expand_shape = list(orig_shape)
            if isinstance(self.axes, int):
                self.axes = (self.axes,)
            for axis in self.axes:
                expand_shape[axis] = 1
        out_grad_expand = out_grad.reshape(expand_shape).broadcast_to(orig_shape)
        node_expand = node.reshape(expand_shape).broadcast_to(orig_shape)

        # then calculate grad
        input_grad = out_grad_expand * exp(input - node_expand)
        return (input_grad,)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
