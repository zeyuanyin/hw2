"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(
            in_features, out_features, nonlinearity="relu"
        ))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, nonlinearity="relu").reshape(
            (1, -1)
        ))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return X.matmul(self.weight) + self.bias.broadcast_to(
            (X.shape[0], self.out_features)
        )
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for i in self.modules:
            x = i.forward(x)
            if x is None:
                raise ValueError("None value is not allowed")
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # print('logits shape:',logits.shape)

        softmax = ops.logsumexp(logits, axes=(1,))
        # .reshape((logits.shape[0], 1)).broadcast_to(logits.shape)
        # print('softmax shape:',softmax.shape)
        # print('y onehot shape:',init.one_hot(logits.shape[1],y).shape)
        y_label = logits * init.one_hot(logits.shape[1], y).sum(axes=(1,))
        y_label = y_label.sum(axes=(1,))
        # print('y_label shape:',y_label.shape)
        res=(softmax - y_label).sum() / logits.shape[0]
        # print('res type',res.data.dtype)
        return res

        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype).reshape((1,self.dim)))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype).reshape((1,self.dim)))

        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype).reshape((1,self.dim))
        self.running_var = init.ones(self.dim, device=device, dtype=dtype).reshape((1,self.dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION

        if self.training:
            mean = x.sum(axes=(0,))/x.shape[0]
            self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            self.running_mean= self.running_mean.reshape((self.dim,))
            mean=mean.reshape((1,x.shape[1])).broadcast_to(x.shape)


            var = ((x-mean)**2).sum(axes=(0,))/x.shape[0]
            self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var
            self.running_var= self.running_var.reshape((self.dim,))
            var=var.reshape((1,x.shape[1])).broadcast_to(x.shape)


            x_norm = (x - mean) / (var + self.eps)**0.5


            return x_norm * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)

        else:
            x_norm = (x - self.running_mean) / (self.running_var + self.eps)**0.5

            return x_norm * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype).reshape((1,self.dim)))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype).reshape((1,self.dim)))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION

        # print("x shape:", x.shape)
        # print("weight shape:", self.weight.shape)
        # print("bias shape:", self.bias.shape)
        # print("dim:", self.dim)
        # print("-----")
        # x.shape = (batch_size, dim)
        Expection = x.sum(axes=1) / x.shape[1]
        Expection = Expection.reshape((x.shape[0], 1)).broadcast_to(x.shape)

        Variance = (ops.power_scalar((x - Expection) ,2)).sum(axes=1) / x.shape[1]
        Variance = Variance.reshape((x.shape[0], 1)).broadcast_to(x.shape)

        x_norm = (x - Expection) / ops.power_scalar((Variance + self.eps) , 0.5)

        return self.weight.broadcast_to(x.shape) * x_norm + self.bias.broadcast_to(x.shape)

        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION

        # Bernoulli distribution
        if self.training:
            mask = init.randb(*x.shape,p=1-self.p)
            return x * mask/(1-self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn.forward(x)
        ### END YOUR SOLUTION
