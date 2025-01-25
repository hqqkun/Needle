"""The module.
"""

import math
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np

from needle.ops.ops_mathematic import tanh, relu
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1 + ops.exp(-x)) ** -1
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        value = 1 / math.sqrt(hidden_size)
        self.W_ih = Parameter(
            init.rand(
                input_size,
                hidden_size,
                low=-value,
                high=value,
                dtype=dtype,
                device=device,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                hidden_size,
                low=-value,
                high=value,
                dtype=dtype,
                device=device,
            )
        )
        self.bias_ih = (
            Parameter(
                init.rand(
                    hidden_size, low=-value, high=value, dtype=dtype, device=device
                )
            )
            if bias
            else None
        )
        self.bias_hh = (
            Parameter(
                init.rand(
                    hidden_size, low=-value, high=value, dtype=dtype, device=device
                )
            )
            if bias
            else None
        )
        self.nonlinearity = tanh if nonlinearity == "tanh" else relu
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION)
        bs, _ = X.shape
        if h is None:
            h = init.zeros(bs, self.hidden_size, dtype=X.dtype, device=X.device)

        h_ = X @ self.W_ih + h @ self.W_hh

        if self.bias_ih:
            h_ = h_ + self.bias_ih.broadcast_to(h_.shape)
        if self.bias_hh:
            h_ = h_ + self.bias_hh.broadcast_to(h_.shape)

        return self.nonlinearity(h_)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cells = [
            RNNCell(
                input_size=input_size if layer == 0 else hidden_size,
                hidden_size=hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                device=device,
                dtype=dtype,
            )
            for layer in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        if h0 is None:
            h0 = init.zeros(
                self.num_layers, bs, self.hidden_size, dtype=X.dtype, device=X.device
            )
        list_X = tuple(ops.split(X, axis=0))  # tuple of (bs, input_size)
        list_h = list(ops.split(h0, axis=0))  # list of (bs, hidden_size)
        output = []

        for time in range(seq_len):
            input = list_X[time]
            tmp_list_h = []
            for layer in range(self.num_layers):
                h_out = self.rnn_cells[layer](input, list_h[layer])
                input = h_out
                tmp_list_h.append(h_out)
            list_h = tmp_list_h
            output.append(list_h[-1])

        # now output is a list of (bs, hidden_size)
        output = ops.stack(output, axis=0)  # (seq_len, bs, hidden_size)
        list_h = ops.stack(list_h, axis=0)  # (num_layers, bs, hidden_size)
        return (output, list_h)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(
        self, input_size, hidden_size, bias=True, device=None, dtype="float32"
    ):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        value = 1 / math.sqrt(hidden_size)
        self.W_ih = Parameter(
            init.rand(
                input_size,
                4 * hidden_size,
                low=-value,
                high=value,
                dtype=dtype,
                device=device,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                4 * hidden_size,
                low=-value,
                high=value,
                dtype=dtype,
                device=device,
            )
        )
        self.bias_ih = (
            Parameter(
                init.rand(
                    4 * hidden_size, low=-value, high=value, dtype=dtype, device=device
                )
            )
            if bias
            else None
        )
        self.bias_hh = (
            Parameter(
                init.rand(
                    4 * hidden_size, low=-value, high=value, dtype=dtype, device=device
                )
            )
            if bias
            else None
        )
        self.sigmoid = Sigmoid()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs, _ = X.shape

        if h is None:
            h0 = init.zeros(bs, self.hidden_size, dtype=X.dtype, device=X.device)
            c0 = init.zeros(bs, self.hidden_size, dtype=X.dtype, device=X.device)
        else:
            h0, c0 = h

        h = X @ self.W_ih + h0 @ self.W_hh
        if self.bias_hh:
            h = h + self.bias_hh.broadcast_to(h.shape)
        if self.bias_ih:
            h = h + self.bias_ih.broadcast_to(h.shape)

        split_h = tuple(ops.split(h, axis=1))
        gates = []
        for i in range(4):
            gates.append(
                ops.stack(
                    split_h[i * self.hidden_size : (i + 1) * self.hidden_size], axis=1
                )
            )
        assert len(gates) == 4
        i = self.sigmoid(gates[0])
        f = self.sigmoid(gates[1])
        g = tanh(gates[2])
        o = self.sigmoid(gates[3])

        c = f * c0 + i * g
        h = o * tanh(c)
        return (h, c)
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = [
            LSTMCell(
                input_size=input_size if layer == 0 else hidden_size,
                hidden_size=hidden_size,
                bias=bias,
                device=device,
                dtype=dtype,
            )
            for layer in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            c_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        if h is None:
            h0 = init.zeros(
                self.num_layers, bs, self.hidden_size, dtype=X.dtype, device=X.device
            )
            c0 = init.zeros(
                self.num_layers, bs, self.hidden_size, dtype=X.dtype, device=X.device
            )
        else:
            h0, c0 = h
        list_X = tuple(ops.split(X, axis=0))  # tuple of (bs, input_size)
        list_h = list(ops.split(h0, axis=0))  # list of (bs, hidden_size)
        list_c = list(ops.split(c0, axis=0))  # list of (bs, hidden_size)
        output = []

        for time in range(seq_len):
            input = list_X[time]
            tmp_lst_h = []
            tmp_lst_c = []
            for layer in range(self.num_layers):
                ht, ct = self.lstm_cells[layer](input, (list_h[layer], list_c[layer]))
                input = ht
                tmp_lst_h.append(ht)
                tmp_lst_c.append(ct)
            list_h = tmp_lst_h
            list_c = tmp_lst_c
            output.append(list_h[-1])

        output = ops.stack(output, axis=0)  # (seq_len, bs, hidden_size)
        list_h = ops.stack(list_h, axis=0)
        list_c = ops.stack(list_c, axis=0)
        return (output, (list_h, list_c))
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
