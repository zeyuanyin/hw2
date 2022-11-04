import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)


class ResidualBlock_(nn.Module):
    def __init__(self, dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
        super().__init__()
        self.residual = nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim),
            )
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.residual(x))


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return ResidualBlock_(dim, hidden_dim, norm, drop_prob)
    ### END YOUR SOLUTION


class MLPResNet_(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=100,
        num_blocks=3,
        num_classes=10,
        norm=nn.BatchNorm1d,
        drop_prob=0.1,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            *[
                ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob)
                for _ in range(num_blocks)
            ],
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return MLPResNet_(dim, hidden_dim, num_blocks, num_classes, norm, drop_prob)
    ### END YOUR SOLUTION


# Returns the average error rate (changed from accuracy) (as a float) and the average loss over all samples (as a float).
def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION

    if opt is None:
        model.eval()
    else:
        model.train()

    total_error = 0
    total_loss = 0
    num=0
    for i, (x, y) in enumerate(dataloader):
        # print('x shape: ', x.shape)# (250, 28, 28, 1)
        x = x.reshape((x.shape[0], -1))
        # print('x shape: ', x.shape) # (250, 784)
        # print('y shape: ', y.shape) # (250,)
        y = y.reshape((y.shape[0], -1))
        # print('y shape: ', y.shape)  #(250, 1)
        if opt is not None:
            opt.reset_grad()

        y_pred = model(x)
        loss = nn.SoftmaxLoss()(y_pred, y)

        # y_hat = y_pred.numpy().argmax(axis=1)
        # print(y_hat!=y.reshape(-1).numpy())
        # print()
        total_error += (y_pred.numpy().argmax(axis=1) != y.reshape(-1).numpy()).sum()
        total_loss += loss.numpy()

        if opt is not None:
            loss.backward()
            opt.step()

        num+=y.shape[0]

    return total_error / num, total_loss / (i + 1)


    ### END YOUR SOLUTION


# Returns a tuple of the training accuracy, training loss, test accuracy, test loss computed in the last epoch of training.
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
    train_dataset = ndl.data.MNISTDataset(
        data_dir + "/train-images-idx3-ubyte.gz",
        data_dir + "/train-labels-idx1-ubyte.gz",
    )
    train_dataloader = ndl.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset = ndl.data.MNISTDataset(
        data_dir + "/t10k-images-idx3-ubyte.gz", data_dir + "/t10k-labels-idx1-ubyte.gz"
    )
    test_dataloader = ndl.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    model = MLPResNet(784, hidden_dim=hidden_dim, num_blocks=3, num_classes=10)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    # train
    for epoch_i in range(epochs):
        train_error, train_loss = epoch(train_dataloader, model, opt)

    # test
    test_error, test_loss = epoch(train_dataloader, model)

    return train_error, train_loss, test_error, test_loss

    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
