"""The module.
"""

import math
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(
                fan_in=in_channels * self.kernel_size * self.kernel_size,
                fan_out=out_channels * self.kernel_size * self.kernel_size,
                shape=(self.kernel_size, self.kernel_size, in_channels, out_channels),
                dtype=dtype,
                device=device,
            )
        )
        value = 1 / math.sqrt(in_channels * (self.kernel_size**2))
        self.bias = Parameter(
            init.rand(out_channels, low=-value, high=value, dtype=dtype, device=device)
            if bias
            else None
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        _, C, H, W = x.shape
        assert C == self.in_channels, "Input channel mismatch"
        assert H == W, "Only square inputs supported"

        x = ops.transpose(x, axes=(1, 2))
        x = ops.transpose(x, axes=(2, 3))

        out = ops.conv(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias:
            out = out + ops.broadcast_to(self.bias, out.shape)

        out = ops.transpose(out, axes=(2, 3))
        out = ops.transpose(out, axes=(1, 2))
        return out
        ### END YOUR SOLUTION
