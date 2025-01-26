"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time

device = ndl.cpu()


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filename, "rb") as file_image:
        _, nums, row, col = struct.unpack(">IIII", file_image.read(16))
        images = np.reshape(
            np.frombuffer(file_image.read(), dtype=np.uint8), (nums, row * col)
        )

    with gzip.open(label_filename, "rb") as file_label:
        _, _ = struct.unpack(">II", file_label.read(8))
        labels = np.frombuffer(file_label.read(), dtype=np.uint8)

    return (np.astype(images / 255.0, np.float32), labels)
    ### END YOUR SOLUTION


def softmax_loss(Z: ndl.Tensor, y_one_hot: ndl.Tensor):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    sum_exp_Z = ndl.exp(Z).sum(axes=(1,))  # (batch_size, )
    value_y = (Z * y_one_hot).sum(axes=(1,))  # (batch_size, )
    sum_error = (ndl.log(sum_exp_Z) - value_y).sum()
    return sum_error / Z.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION

    # get y_one_hot
    y_one_hot = np.zeros((y.shape[0], W2.shape[1]), dtype=np.float32)
    y_one_hot[np.arange(len(y)), y] = 1.0

    rows = np.shape(X)[0]
    num_iter = math.ceil(rows / batch)
    for i in range(num_iter):
        end = min(rows, (i + 1) * batch)
        num = end - i * batch
        batchX = ndl.Tensor(X[i * batch : end])
        batchY = ndl.Tensor(y_one_hot[i * batch : end])
        Z = ndl.relu(batchX.matmul(W1)).matmul(W2)
        loss: ndl.Tensor = softmax_loss(Z, batchY)
        loss.backward()
        W1 = ndl.Tensor(W1.data - lr * W1.grad.data)
        W2 = ndl.Tensor(W2.data - lr * W2.grad.data)

    return (W1, W2)
    ### END YOUR SOLUTION


### CIFAR-10 training ###
def epoch_general_cifar10(
    dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None, device=None
):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()

    totalLoss = 0.0
    acc = 0.0
    for batch in dataloader:
        X, y = batch
        X, y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
        logits: ndl.Tensor = model(X)
        loss: ndl.Tensor = loss_fn(logits, y)
        labels = np.argmax(logits.numpy(), axis=1)
        acc += np.sum(labels == y.numpy())
        totalLoss += loss.data.numpy() * y.shape[0]

        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()
    numExamples = len(dataloader.dataset)
    return (acc / numExamples, totalLoss / numExamples)
    ### END YOUR SOLUTION


def train_cifar10(
    model,
    dataloader,
    n_epochs=1,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    loss_fn=nn.SoftmaxLoss(),
    device=None,
):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cifar10(
            dataloader, model, loss_fn=loss_fn, opt=opt, device=device
        )
        print(f"Epoch: {epoch}, Acc: {avg_acc}, Loss: {avg_loss}")
    return (avg_acc, avg_loss)
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss(), device=None):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_cifar10(
        dataloader, model, loss_fn=loss_fn, device=device
    )
    print(f"Evaluation Acc: {avg_acc}, Evaluation Loss: {avg_loss}")
    return (avg_acc, avg_loss)
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(
    data,
    model,
    seq_len=40,
    loss_fn=nn.SoftmaxLoss(),
    opt=None,
    clip=None,
    device=None,
    dtype="float32",
):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()

    numExamples = 0
    totalLoss = 0.0
    acc = 0.0

    nbatch, _ = data.shape
    hidden = None
    for i in range(0, nbatch - 1, seq_len):
        X, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        numExamples += y.shape[0]
        logits, hidden = model(X, hidden)

        loss = loss_fn(logits, y)

        labels = np.argmax(logits.numpy(), axis=1)
        acc += np.sum(labels == y.numpy())
        totalLoss += loss.data.numpy() * y.shape[0]

        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()

    return (acc / numExamples, totalLoss / numExamples)

    ### END YOUR SOLUTION


def train_ptb(
    model,
    data,
    seq_len=40,
    n_epochs=1,
    optimizer=ndl.optim.SGD,
    lr=4.0,
    weight_decay=0.0,
    loss_fn=nn.SoftmaxLoss(),
    clip=None,
    device=None,
    dtype="float32",
):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(
            data,
            model,
            seq_len=seq_len,
            loss_fn=loss_fn,
            opt=opt,
            clip=clip,
            device=device,
            dtype=dtype,
        )
        print(f"Epoch: {epoch}, Acc: {avg_acc}, Loss: {avg_loss}")
    return (avg_acc, avg_loss)
    ### END YOUR SOLUTION


def evaluate_ptb(
    model, data, seq_len=40, loss_fn=nn.SoftmaxLoss(), device=None, dtype="float32"
):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_ptb(
        data, model, seq_len=seq_len, loss_fn=loss_fn, device=device, dtype=dtype
    )
    print(f"Evaluation Acc: {avg_acc}, Evaluation Loss: {avg_loss}")
    return (avg_acc, avg_loss)
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
