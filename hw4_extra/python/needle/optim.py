"""Optimization module"""

import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for w in self.params:
            if self.weight_decay > 0:
                grad = (w.grad + self.weight_decay * w.data).data
            else:
                grad = w.grad.data
            if w in self.u:
                self.u[w] = (
                    self.momentum * self.u[w] + (1 - self.momentum) * grad
                ).detach()
            else:
                self.u[w] = ((1 - self.momentum) * grad).detach()
            w.data -= ndl.Tensor(self.lr * self.u[w], dtype=w.dtype)
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t = self.t + 1
        for w in self.params:
            if w in self.m:
                m = (
                    self.beta1 * self.m[w]
                    + (1 - self.beta1) * (self.weight_decay * w.data + w.grad).data
                )
            else:
                m = ((1 - self.beta1) * (self.weight_decay * w.data + w.grad)).data
            if w in self.v:
                v = (
                    self.beta2 * self.v[w]
                    + (1 - self.beta2)
                    * ((self.weight_decay * w.data + w.grad) ** 2).data
                )
            else:
                v = (
                    (1 - self.beta2) * ((self.weight_decay * w.data + w.grad) ** 2)
                ).data
            self.m[w] = m
            self.v[w] = v
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            w.data -= ndl.Tensor(
                self.lr * (m_hat / (v_hat**0.5 + self.eps)),
                dtype=w.dtype,
            )
        ## END YOUR SOLUTION