import sys

sys.path.append("./python")
import needle as ndl
import needle.nn as nn
import numpy as np

np.random.seed(0)


def ConvBN(a, b, k, s, device=None):
    return nn.Sequential(
        nn.Conv(a, b, k, s, device=device),
        nn.BatchNorm2d(dim=b, device=device),
        nn.ReLU(),
    )


def ResidualBlock(a, b, k, s, device=None):
    return nn.Residual(
        nn.Sequential(
            ConvBN(a, b, k, s, device=device),
            ConvBN(a, b, k, s, device=device),
        )
    )


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.model = nn.Sequential(
            ConvBN(3, 16, 7, 4, device=device),
            ConvBN(16, 32, 3, 2, device=device),
            ResidualBlock(32, 32, 3, 1, device=device),
            ConvBN(32, 64, 3, 2, device=device),
            ConvBN(64, 128, 3, 2, device=device),  #! THIS is not 32
            ResidualBlock(128, 128, 3, 1, device=device),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=128, device=device),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10, device=device),
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.model(x)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(
        self,
        embedding_size,
        output_size,
        hidden_size,
        num_layers=1,
        seq_model="rnn",
        seq_len=40,
        device=None,
        dtype="float32",
    ):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            num_embeddings=output_size,
            embedding_dim=embedding_size,
            device=device,
            dtype=dtype,
        )
        
        self.seq_model = None
        
        if seq_model == "rnn":
            self.seq_model = nn.RNN(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            device=device,
            dtype=dtype,
        )
        elif seq_model == "lstm":
            self.seq_model = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            device=device,
            dtype=dtype,
        )
        elif seq_model == "transformer":
            self.seq_model = nn.Transformer(
                embedding_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                device=device,
                dtype=dtype,
                sequence_len=seq_len,
            )
        else:
            raise ValueError("seq_model must be 'rnn' or 'lstm' or 'transformer'")

        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
            device=device,
            dtype=dtype,
        )

        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        embedded = self.embedding(x)
        out, h = self.seq_model(embedded, h)
        out = out.reshape(shape=(seq_len * bs, self.hidden_size))
        out = self.linear(out)

        return (out, h)
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset(
        "data/cifar-10-batches-py", train=True
    )
    train_loader = ndl.data.DataLoader(
        cifar10_train_dataset, 128, ndl.cpu(), dtype="float32"
    )
    print(cifar10_train_dataset[1][0].shape)
