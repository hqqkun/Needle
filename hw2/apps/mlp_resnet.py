import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    main_block = nn.Sequential(
        nn.Linear(in_features=dim, out_features=hidden_dim),
        norm(dim=hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=drop_prob),
        nn.Linear(in_features=hidden_dim, out_features=dim),
        norm(dim=dim),
    )
    r = nn.Residual(main_block)
    return nn.Sequential(r, nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[
            ResidualBlock(
                dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                norm=norm,
                drop_prob=drop_prob,
            )
            for _ in range(num_blocks)
        ],
        nn.Linear(hidden_dim, num_classes),
    )
    ### END YOUR SOLUTION


def epoch(dataloader: ndl.data.DataLoader, model: nn.Module, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    lossFunc = nn.SoftmaxLoss()
    if opt:
        model.train()
    else:
        model.eval()

    totalLoss = []
    error = 0.0
    for X, y in dataloader:
        logits: nn.Tensor = model(X)
        loss: nn.Tensor = lossFunc(logits, y)
        labels = np.argmax(logits.numpy(), axis=1)
        error += np.sum(labels != y.numpy())
        totalLoss.append(loss.numpy())

        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()

    numExamples = len(dataloader.dataset)
    return (error / numExamples, np.mean(totalLoss))

    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    resnet = MLPResNet(dim=28 * 28, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    train_set = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz",
        f"{data_dir}/train-labels-idx1-ubyte.gz",
    )
    test_set = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz"
    )

    train_loader = ndl.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(test_set, batch_size=batch_size)
    for _ in range(epochs):
        train_err, train_loss = epoch(train_loader, resnet, opt=opt)
    test_err, test_loss = epoch(test_loader, resnet, opt=None)
    return (train_err, train_loss, test_err, test_loss)
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
