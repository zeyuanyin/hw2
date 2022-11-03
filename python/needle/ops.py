"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)

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


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return pow(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        a = node.inputs[0]
        return self.scalar * a**( self.scalar - 1) * out_grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, -out_grad * a / b**2
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:  # defaults to the last two axes
            return array_api.swapaxes(a, -2, -1)
        else:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        if self.axes is None:
            return transpose(out_grad, axes=(-2, -1))
        else:
            return transpose(out_grad, axes=self.axes)

        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return reshape(out_grad, a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # print(a.shape, "->", self.shape)
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION


        #TODO: this is not correct
        # a = node.inputs[0]

        # if len(a.shape) != len(self.shape):
        #     reshape_out_grad = out_grad.sum(
        #         tuple(range(0, len(self.shape) - len(a.shape)))
        #     )
        # else:
        #     reshape_out_grad = out_grad  # reshape_out_grad.shape = a.shape

        # axis = None
        # for x, y in zip(a.shape, self.shape):
        #     if x != y:
        #         axis = a.shape.index(x)
        #         break

        # print('-----------------')
        # print("axis:", axis)
        # print("a.shape:", a.shape)
        # print("self.shape:", self.shape)
        # print("reshape_out_grad.shape:", reshape_out_grad.shape)
        # if axis is None:
        #     return reshape_out_grad
        # else:
        #     return reshape_out_grad.sum(axes=(axis,)).reshape(a.shape)

        input_shape = node.inputs[0].shape

        if len(input_shape) != len(self.shape):
            input_shape = tuple(
                [1] * (len(self.shape) - len(input_shape)) + list(input_shape)
            )

        axes = tuple(i for i, a in enumerate(input_shape) if a == 1)
        return (out_grad.sum(axes).reshape(input_shape),)


        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # print(a.shape, "->", array_api.sum(a, self.axes).shape, "by axes", self.axes)
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        if self.axes is None:
            return broadcast_to(out_grad, a.shape)
        else:
            new_shape = list(a.shape)

            if isinstance(self.axes, int):
                new_shape[self.axes] = 1
            else:
                for axis in self.axes:
                    new_shape[axis] = 1

            new_shape = tuple(new_shape)
            reshaped_out_grad = reshape(out_grad, new_shape)

            return reshaped_out_grad.broadcast_to(a.shape)

        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        a, b = node.inputs
        res_a = out_grad @ b.transpose()
        res_b = a.transpose() @ out_grad

        def reverse_broadcast(x, y):  # x has the shape which is broadcasted from y
            if x.shape == y.shape:
                return x
            else:
                return x.sum(tuple(range(0, len(x.shape) - len(y.shape))))

        return reverse_broadcast(res_a, a), reverse_broadcast(res_b, b)

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
        return -1 * out_grad
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
        a = node.inputs[0]
        # print(a.shape, out_grad.shape)
        # print(type(a),type(out_grad))
        return Tensor(1 / a.numpy()) * out_grad
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
        a = node.inputs[0]
        return exp(a) * out_grad
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

        a = node.inputs[0].realize_cached_data()

        return out_grad * Tensor(a > 0)

        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION

        Z_max = array_api.max(Z, self.axes)
        # print(type(Z))
        # print('self.axes:', self.axes)
        # print('Z:', Z)
        # print('shape:',Z.shape)
        # print('Z_max:', Z_max)

        # find reshape shape original (3,4,5)  axes=(1,2) -> max (3,) -> reshape (3,1,1)
        if self.axes is None: #(3,4,5) axes=(1,2)  -> (1,) reshape_shape = (1,1,1)
            reshape_shape = list(1 for _ in range(len(Z.shape)))
            reshape_shape = tuple(reshape_shape)
        else:
            reshape_shape = list(Z.shape)
            for axis in self.axes:
                reshape_shape[axis] = 1
            reshape_shape = tuple(reshape_shape)
        Z_max_ori = Z_max.reshape(reshape_shape)
        Z_max_ori=array_api.broadcast_to(Z_max_ori, Z.shape)

        return Z_max + array_api.log(array_api.sum(array_api.exp(Z - Z_max_ori), self.axes))

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        Z = node.inputs[0]
        Z_max = array_api.max(Z.cached_data, self.axes)

        Z_max=Tensor(Z_max)

        if self.axes is None:
            reshape_shape = list(1 for _ in range(len(Z.shape)))
            reshape_shape = tuple(reshape_shape)
        else:
            reshape_shape = list(Z.shape)
            for axis in self.axes:
                reshape_shape[axis] = 1
            reshape_shape = tuple(reshape_shape)
        Z_max_ori = Z_max.reshape(reshape_shape)
        Z_max_ori=Z_max_ori.broadcast_to(Z.shape)


        # print('self.axes:', self.axes)
        # print('Z:', Z)
        # print('shape:',Z.shape)
        # print('Z_max:', Z_max)
        # print('grad:', out_grad.shape)
        Z_exp = exp(Z - Z_max_ori)
        sum_exp=summation(Z_exp, self.axes)
        sum_exp=sum_exp.reshape(reshape_shape)
        sum_exp=sum_exp.broadcast_to(Z.shape)

        out_grad = out_grad.reshape(reshape_shape)
        out_grad = out_grad.broadcast_to(Z.shape)


        return out_grad * Z_exp / sum_exp
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
