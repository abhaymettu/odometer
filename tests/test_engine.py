"""Gradients of every differentiable op against central finite differences, and the meter's arithmetic."""

import numpy as np
import pytest

import odometer as od
from odometer import _engine


@pytest.fixture(autouse=True)
def fresh_meter():
    _engine._M.reset(None, 0)


def leaf(shape, seed, positive=False):
    a = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    if positive:
        a = np.abs(a) + 0.5
    return od.param(od.Tensor(a))


def check(fn, *shapes, positive=False, tol=3e-2):
    xs = [leaf(s, i, positive) for i, s in enumerate(shapes)]
    w = np.random.default_rng(99).standard_normal(fn(*xs).shape).astype(np.float32)
    loss = lambda *a: od.sum(fn(*a) * od.Tensor(w))
    loss(*xs).backward()
    eps = 1e-2
    for x in xs:
        num = np.zeros(x.shape, np.float64)
        it = np.nditer(x._d, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = float(x._d[i])
            x._d[i] = old + eps
            hi = loss(*xs)._d.item()
            x._d[i] = old - eps
            lo = loss(*xs)._d.item()
            x._d[i] = old
            num[i] = (hi - lo) / (2 * eps)
        ana = x.grad._d
        assert np.allclose(ana, num, rtol=tol, atol=tol), (fn, np.abs(ana - num).max())


UNARY = [od.neg, od.exp, od.tanh, od.sigmoid, od.sin, od.cos, lambda x: x**2, lambda x: od.softmax(x),
         lambda x: od.log_softmax(x), lambda x: od.max(x, -1), lambda x: od.mean(x, 0, keepdims=True),
         lambda x: od.sum(x, (0, 1)), lambda x: x.reshape(6, 2).T, lambda x: x.transpose(1, 0, 2),
         lambda x: x[1:, ::2], lambda x: od.split(x, 2, -1)[1], lambda x: od.tril(x), od.relu, od.abs]


@pytest.mark.parametrize("fn", UNARY)
def test_unary(fn):
    check(fn, (3, 2, 2))


@pytest.mark.parametrize("fn", [od.log, od.sqrt, od.rsqrt, lambda x: 1.0 / x])
def test_positive_domain(fn):
    check(fn, (3, 4), positive=True)


@pytest.mark.parametrize("fn", [od.add, od.sub, od.mul, od.maximum, od.minimum,
                                lambda a, b: od.concat([a, b], 0), lambda a, b: od.where(a > 0, a, b)])
def test_binary(fn):
    check(fn, (3, 4), (3, 4))


@pytest.mark.parametrize("fn", [od.add, od.mul, od.sub])
def test_broadcast(fn):
    check(fn, (2, 3, 4), (4,))


def test_div():
    xs = [leaf((3, 4), 0), leaf((3, 4), 1, positive=True)]
    check(lambda a, b: a / (b * b + 1.0), (3, 4), (3, 4))


def test_matmul_2d_weight():
    check(od.matmul, (2, 3, 4), (4, 5))


def test_matmul_batched():
    check(od.matmul, (2, 3, 4), (2, 4, 3))


def test_cross_entropy_and_embedding():
    targets = od.Tensor(np.array([[1, 3, 0], [2, 2, 4]], np.int32))
    idx = od.Tensor(np.array([[0, 2, 1], [1, 1, 0]], np.int32))
    check(lambda logits: od.cross_entropy(logits, targets) * od.ones((1,)), (2, 3, 5))
    check(lambda table: od.embedding(table, idx), (3, 4))


def test_costs_are_shape_only():
    a, b = od.ones((8, 16)), od.ones((16, 32))
    start = od.used()
    od.matmul(a, b)
    assert od.used() - start == 2000 + 8 * 32 * 64 // 128 + 256
    start = od.used()
    od.exp(a)
    a.reshape(16, 8).T
    assert od.used() - start == 3 * 2000 + 8 * 16


def test_backward_is_charged():
    w = od.param(od.ones((16, 32)))
    x = od.ones((8, 16))
    start = od.used()
    loss = od.sum(od.matmul(x, w))
    fwd = od.used() - start
    loss.backward()
    # two nodes pay the call price in the walk (the leaf pays nothing), then the sum's broadcast and dW pay theirs
    assert od.used() - start - fwd == 2 * 2000 + (2000 + 8 * 32 // 8) + (2000 + 16 * 32 * 64 // 128 + 256)


def test_budget_is_sticky():
    _engine._M.reset(3000, 0)
    od.ones((10, 10))
    with pytest.raises(od.BudgetExhausted):
        od.ones((10, 10))
    with pytest.raises(od.BudgetExhausted):
        od.ones((1,))


def test_tensor_hides_values():
    t = od.ones((2, 2))
    assert "1" not in f"{t}".replace("shape=(2, 2)", "")
    for leak in (bool, float, int, list, iter):
        with pytest.raises(od.RuleViolation):
            leak(t)
