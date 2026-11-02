"""The metered tensor engine.

Every engine call advances one global odometer by a price that is a function of tensor shapes only, never of the
machine or the values, so the same program reaches the same reading everywhere.

The unit is a tick, priced to be about one nanosecond of one laptop core. tools/calibrate.py measures nanoseconds per
tick for every op on real machines and fails if any op drifts more than a factor of 5 from the median, so an entry
cannot get cheap work by picking mispriced ops.

Prices (v3), per call:
  * every call: 2000 ticks, the Python overhead of the call itself. Backward passes pay it again per node.
  * matmul of (..., m, k) @ (..., k, n): m * max(k, 64) * n / 128 plus 256, per batch element (a 2-D right side is one
    flat call, so m counts the batch and the 256 is paid once). The same again per input needing a gradient, with the
    inner and outer dimensions swapped to match the product backward actually does.
  * cheap elementwise work (add, mul, negate, compare, copy, sum): 1 tick per 4 elements. ones and full 1 per 8, and
    zeros and the backward of a sum 1 per 16 and 1 per 8, because they only fill memory.
  * sqrt, div, tanh: 1 tick per 2 elements. exp, log, sigmoid, sin, cos: 1 per element. where, power, sign: 2 per
    element. Integer % and //: 1 per element.
  * randn 4 and rand 2 per element. Logit ops (softmax, log_softmax, cross_entropy): 2 per logit, 1 per logit backward;
    cross_entropy pays the call price twice.
  * embedding: 1 per 4 elements, plus per row 6 and 1 per million table floats; backward 16 per element plus the zeroed
    table.
  * concat: 256 per input tensor on top of the elements, 1024 per input tensor backward. split: 2000 per part forward,
    512 per part backward. data.batch: 2000 plus 64 per row plus 1 per byte.
  * views (transpose, slicing, a reshape that keeps the memory) are free apart from the call price. A reshape that has
    to copy pays 1 per element.

When a charge would pass the budget, the op raises BudgetExhausted and every later op raises too.
"""

import contextlib as _contextlib
import math as _math

import numpy as _np

_F = _np.float32
_I = _np.int32


class BudgetExhausted(Exception):
    pass


class RuleViolation(Exception):
    pass


class _Meter:
    def __init__(self):
        self.reset(None, 0)

    def reset(self, budget, seed, mode="train", item_cap=20_000):
        self.budget = budget
        self.used = 0
        self.exhausted = False
        self.mode = mode
        self.grad = mode == "train"
        self.items = 0
        self.item_cap = item_cap
        self.init_rng = _np.random.default_rng([seed, 0])
        self.data_rng = _np.random.default_rng([seed, 1])


_M = _Meter()


_CALL = 2000  # ticks for one engine call, whatever its size


def _tick(n=0):
    if _M.exhausted:
        raise BudgetExhausted("the odometer already reached its budget")
    n = int(n) if n > 0 else 0  # a negative price would run the odometer backwards
    n += _CALL
    if _M.budget is not None and _M.used + n > _M.budget:
        _M.exhausted = True
        raise BudgetExhausted(f"budget {_M.budget:.3e} reached at {_M.used:.3e}")
    _M.used += n


def _train_only(what):
    if _M.mode != "train":
        raise RuleViolation(f"{what} is not allowed during evaluation")


def used():
    return _M.used


def budget():
    return _M.budget


def remaining():
    return None if _M.budget is None else _M.budget - _M.used


@_contextlib.contextmanager
def no_grad():
    prev = _M.grad
    _M.grad = False
    try:
        yield
    finally:
        _M.grad = prev


class Tensor:
    """An opaque array. Values leave the engine only through item(), which is rationed."""

    __slots__ = ("_d", "_prev", "_back", "_rg", "grad")

    def __init__(self, d, prev=(), back=None, rg=False):
        self._d = d
        self._prev = prev
        self._back = back
        self._rg = rg
        self.grad = None

    # ---- what an entry may know about a tensor
    @property
    def shape(self):
        return self._d.shape

    @property
    def ndim(self):
        return self._d.ndim

    @property
    def size(self):
        return int(self._d.size)

    @property
    def is_int(self):
        return self._d.dtype.kind == "i"

    @property
    def requires_grad(self):
        return self._rg

    def __len__(self):
        return self._d.shape[0]

    def __repr__(self):
        kind = "int" if self.is_int else "float"
        return f"Tensor(shape={self.shape}, {kind})"

    __str__ = __repr__

    def __format__(self, spec):
        return repr(self)

    def __bool__(self):
        raise RuleViolation("a tensor has no truth value; use item()")

    def __float__(self):
        raise RuleViolation("use item() to read a scalar")

    __int__ = __index__ = __float__

    def __iter__(self):
        raise RuleViolation("tensors are not iterable")

    def __array__(self, *args, **kwargs):
        raise RuleViolation("tensors cannot be converted to arrays")

    __hash__ = object.__hash__

    def item(self):
        _train_only("item()")
        if self._d.size != 1:
            raise RuleViolation("item() needs a one-element tensor")
        _M.items += 1
        if _M.items > _M.item_cap:
            raise RuleViolation(f"more than {_M.item_cap} item() reads")
        _tick()
        return self._d.item()

    def detach(self):
        return Tensor(self._d)

    # ---- in-place update of a parameter (optimizers)
    def update(self, value):
        _train_only("update()")
        value = _as_tensor(value, self)
        if value.shape != self.shape:
            value = Tensor(_np.broadcast_to(value._d, self.shape))
        _tick(self.size // 4)
        self._d[...] = value._d

    # ---- autodiff
    def backward(self):
        _train_only("backward()")
        if self.size != 1:
            raise RuleViolation("backward() needs a scalar loss")
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if id(node) in seen or not node._rg:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._prev:
                stack.append((p, False))
        grads = {id(self): _np.ones_like(self._d)}
        for node in reversed(order):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node._back is None:
                if node.grad is None:
                    node.grad = Tensor(g.astype(_F, copy=False))
                else:
                    _tick(g.size // 4)
                    node.grad._d = node.grad._d + g
                continue
            _tick()
            need = [p._rg for p in node._prev]
            for p, pg in zip(node._prev, node._back(g, need)):
                if pg is None or not p._rg:
                    continue
                if id(p) in grads:
                    _tick(pg.size // 4)
                    grads[id(p)] = grads[id(p)] + pg
                else:
                    grads[id(p)] = pg
            node._prev, node._back = (), None

    # ---- operators
    def __add__(self, o):
        return add(self, o)

    def __radd__(self, o):
        return add(o, self)

    def __sub__(self, o):
        return sub(self, o)

    def __rsub__(self, o):
        return sub(o, self)

    def __mul__(self, o):
        return mul(self, o)

    def __rmul__(self, o):
        return mul(o, self)

    def __truediv__(self, o):
        return div(self, o)

    def __rtruediv__(self, o):
        return div(o, self)

    def __neg__(self):
        return neg(self)

    def __pow__(self, p):
        return power(self, p)

    def __matmul__(self, o):
        return matmul(self, o)

    def __mod__(self, o):
        return _int_binary(self, o, _np.mod, per_4=4)

    def __floordiv__(self, o):
        return _int_binary(self, o, _np.floor_divide, per_4=4)

    def __gt__(self, o):
        return _compare(self, o, _np.greater)

    def __ge__(self, o):
        return _compare(self, o, _np.greater_equal)

    def __lt__(self, o):
        return _compare(self, o, _np.less)

    def __le__(self, o):
        return _compare(self, o, _np.less_equal)

    def __getitem__(self, key):
        return getitem(self, key)

    def reshape(self, *shape):
        return reshape(self, *shape)

    def transpose(self, *axes):
        return transpose(self, *axes)

    def swapaxes(self, a, b):
        return swapaxes(self, a, b)

    def sum(self, axis=None, keepdims=False):
        return sum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return mean(self, axis, keepdims)

    def max(self, axis=None, keepdims=False):
        return max(self, axis, keepdims)

    def float(self):
        return to_float(self)

    @property
    def T(self):
        return swapaxes(self, -1, -2)


# ---------------------------------------------------------------- helpers


def _as_tensor(x, like=None):
    if isinstance(x, Tensor):
        return x
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise RuleViolation(f"expected a tensor or a number, got {type(x).__name__}")
    dtype = _I if (like is not None and like.is_int and isinstance(x, int)) else _F
    return Tensor(_np.asarray(x, dtype=dtype))


def _make(d, parents, back):
    rg = _M.grad and any(p._rg for p in parents)
    return Tensor(d, parents if rg else (), back if rg else None, rg)


def _unbroadcast(g, shape):
    if g.shape == shape:
        return g
    _tick(g.size // 4)
    while g.ndim > len(shape):
        g = g.sum(0)
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and g.shape[i] != 1)
    return g.sum(axes, keepdims=True) if axes else g


def _floats(*ts):
    for t in ts:
        if t.is_int:
            raise RuleViolation("this op needs float tensors; convert with .float()")


# ---------------------------------------------------------------- creation


def param(x):
    """Mark a tensor as a trainable leaf."""
    _train_only("param()")
    _tick(x.size // 4)
    t = Tensor(_np.array(x._d, dtype=_F))
    t._rg = True
    return t


def zeros(shape):
    _tick(_np.prod(shape) // 16)
    return Tensor(_np.zeros(shape, _F))


def ones(shape):
    _tick(_np.prod(shape) // 8)
    return Tensor(_np.ones(shape, _F))


def full(shape, value):
    _tick(_np.prod(shape) // 8)
    return Tensor(_np.full(shape, float(value), _F))


def arange(n):
    _tick(n // 4)
    return Tensor(_np.arange(n, dtype=_F))


def randn(shape, std=1.0):
    _train_only("randn()")
    _tick(4 * _np.prod(shape))
    return Tensor((_M.init_rng.standard_normal(shape, dtype=_F) * _F(std)).astype(_F))


def rand(shape):
    _train_only("rand()")
    _tick(2 * _np.prod(shape))
    return Tensor(_M.init_rng.random(shape, dtype=_F))


def tril(x, k=0):
    """Zero the elements above the k-th diagonal of the last two axes."""
    _floats(x)
    _tick(2 * x.size)
    keep = _np.tril(_np.ones(x.shape[-2:], dtype=bool), k)
    out = _np.where(keep, x._d, _F(0))

    def back(g, need):
        _tick(2 * g.size)
        return (_np.where(keep, g, _F(0)),)

    return _make(out, (x,), back)


def where(mask, a, b):
    """mask is a float tensor of zeros and ones (from a comparison or tril)."""
    a, b = _as_tensor(a), _as_tensor(b)
    _floats(mask, a, b)
    m = mask._d != 0
    out = _np.where(m, a._d, b._d).astype(_F)
    _tick(2 * out.size)

    def back(g, need):
        ga = gb = None
        if need[1]:
            _tick(2 * g.size)
            ga = _unbroadcast(_np.where(m, g, _F(0)), a.shape)
        if need[2]:
            _tick(2 * g.size)
            gb = _unbroadcast(_np.where(m, _F(0), g), b.shape)
        return (None, ga, gb)

    return _make(out, (mask, a, b), back)


# ---------------------------------------------------------------- elementwise


def _unary(x, f, df, fwd=4, bwd=1):
    """df(x_array, out_array, g) -> grad. fwd and bwd are prices in ticks per 4 elements."""
    _floats(x)
    _tick(x.size * fwd // 4)
    xd = x._d
    out = f(xd).astype(_F, copy=False)

    def back(g, need):
        _tick(g.size * bwd // 4)
        return (df(xd, out, g),)

    return _make(out, (x,), back)


def neg(x):
    return _unary(x, _np.negative, lambda x, y, g: -g, fwd=1)


def exp(x):
    return _unary(x, _np.exp, lambda x, y, g: g * y)


def log(x):
    return _unary(x, _np.log, lambda x, y, g: g / x, bwd=2)


def sqrt(x):
    return _unary(x, _np.sqrt, lambda x, y, g: g * _F(0.5) / y, fwd=2, bwd=2)


def rsqrt(x):
    return _unary(x, lambda a: 1 / _np.sqrt(a), lambda x, y, g: g * _F(-0.5) * y * y * y, fwd=2, bwd=2)


def tanh(x):
    return _unary(x, _np.tanh, lambda x, y, g: g * (1 - y * y), fwd=2, bwd=2)


def sigmoid(x):
    return _unary(x, lambda a: 1 / (1 + _np.exp(-a)), lambda x, y, g: g * y * (1 - y), bwd=2)


def relu(x):
    return _unary(x, lambda a: _np.maximum(a, _F(0)), lambda x, y, g: g * (x > 0), fwd=2)


def sin(x):
    return _unary(x, _np.sin, lambda x, y, g: g * _np.cos(x), bwd=4)


def cos(x):
    return _unary(x, _np.cos, lambda x, y, g: -g * _np.sin(x), bwd=4)


def abs(x):
    return _unary(x, _np.abs, lambda x, y, g: g * _np.sign(x), fwd=1, bwd=8)


def sign(x):
    """No gradient."""
    _floats(x)
    _tick(x.size * 2)
    return Tensor(_np.sign(x._d).astype(_F))


def power(x, p):
    if not isinstance(p, (int, float)) or isinstance(p, bool):
        raise RuleViolation("power() takes a number exponent")
    return _unary(x, lambda a: _np.power(a, _F(p)), lambda x, y, g: g * _F(p) * _np.power(x, _F(p - 1)), fwd=8, bwd=8)


def add(a, b):
    a, b = _as_tensor(a, b if isinstance(b, Tensor) else None), _as_tensor(b, a)
    if a.is_int or b.is_int:
        return _int_binary(a, b, _np.add)
    out = a._d + b._d
    _tick(out.size // 4)

    def back(g, need):
        return (_unbroadcast(g, a.shape) if need[0] else None, _unbroadcast(g, b.shape) if need[1] else None)

    return _make(out, (a, b), back)


def sub(a, b):
    a, b = _as_tensor(a, b if isinstance(b, Tensor) else None), _as_tensor(b, a)
    if a.is_int or b.is_int:
        return _int_binary(a, b, _np.subtract)
    out = a._d - b._d
    _tick(out.size // 4)

    def back(g, need):
        gb = None
        if need[1]:
            _tick(g.size // 4)
            gb = _unbroadcast(-g, b.shape)
        return (_unbroadcast(g, a.shape) if need[0] else None, gb)

    return _make(out, (a, b), back)


def mul(a, b):
    a, b = _as_tensor(a, b if isinstance(b, Tensor) else None), _as_tensor(b, a)
    if a.is_int or b.is_int:
        return _int_binary(a, b, _np.multiply)
    ad, bd = a._d, b._d
    out = ad * bd
    _tick(out.size // 4)

    def back(g, need):
        ga = gb = None
        if need[0]:
            _tick(g.size // 4)
            ga = _unbroadcast(g * bd, a.shape)
        if need[1]:
            _tick(g.size // 4)
            gb = _unbroadcast(g * ad, b.shape)
        return (ga, gb)

    return _make(out, (a, b), back)


def div(a, b):
    a, b = _as_tensor(a), _as_tensor(b)
    _floats(a, b)
    ad, bd = a._d, b._d
    out = (ad / bd).astype(_F, copy=False)
    _tick(out.size // 2)

    def back(g, need):
        ga = gb = None
        if need[0]:
            _tick(g.size // 2)
            ga = _unbroadcast(g / bd, a.shape)
        if need[1]:
            _tick(g.size)
            gb = _unbroadcast(-g * out / bd, b.shape)
        return (ga, gb)

    return _make(out, (a, b), back)


def maximum(a, b):
    a, b = _as_tensor(a), _as_tensor(b)
    _floats(a, b)
    out = _np.maximum(a._d, b._d)
    _tick(out.size // 2)
    pick_a = a._d >= b._d

    def back(g, need):
        ga = gb = None
        if need[0]:
            _tick(g.size // 4)
            ga = _unbroadcast(g * pick_a, a.shape)
        if need[1]:
            _tick(g.size // 4)
            gb = _unbroadcast(g * ~pick_a, b.shape)
        return (ga, gb)

    return _make(out, (a, b), back)


def minimum(a, b):
    return neg(maximum(neg(a), neg(b)))


def _compare(a, b, f):
    a, b = _as_tensor(a), _as_tensor(b, a)
    out = f(a._d, b._d).astype(_F)
    _tick(out.size // 4)
    return Tensor(out)


def _int_binary(a, b, f, per_4=1):
    a, b = _as_tensor(a), _as_tensor(b, a)
    if not (a.is_int and b.is_int):
        raise RuleViolation("integer ops need two integer operands")
    out = f(a._d, b._d).astype(_I)
    _tick(out.size * per_4 // 4)
    return Tensor(out)


def to_float(x):
    _tick(x.size // 4)
    return Tensor(x._d.astype(_F))


# ---------------------------------------------------------------- reductions


def _norm_axis(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    axes = axis if isinstance(axis, tuple) else (axis,)
    return tuple(a % ndim for a in axes)


def sum(x, axis=None, keepdims=False):
    _floats(x)
    axes = _norm_axis(axis, x.ndim)
    out = x._d.sum(axis=axes, keepdims=keepdims, dtype=_F)
    _tick(x.size // 4)

    def back(g, need):
        _tick(x.size // 8)
        gk = g if keepdims else _np.expand_dims(g, axes)
        return (_np.broadcast_to(gk, x.shape).astype(_F),)

    return _make(_np.asarray(out, _F), (x,), back)


def mean(x, axis=None, keepdims=False):
    axes = _norm_axis(axis, x.ndim)
    n = int(_np.prod([x.shape[a] for a in axes]))
    return mul(sum(x, axis, keepdims), 1.0 / n)


def max(x, axis=None, keepdims=False):
    _floats(x)
    axes = _norm_axis(axis, x.ndim)
    out = x._d.max(axis=axes, keepdims=True)
    _tick(x.size // 4)

    def back(g, need):
        _tick(x.size)
        gk = g if keepdims else _np.expand_dims(g, axes)
        hit = x._d == out
        return (hit * gk / hit.sum(axis=axes, keepdims=True),)

    return _make(out if keepdims else out.squeeze(axes), (x,), back)


# ---------------------------------------------------------------- shapes (views are free)


def _reshaped(a, shape):
    out = a.reshape(shape)
    _tick(0 if _np.may_share_memory(out, a) else out.size)
    return out


def reshape(x, *shape):
    if len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]
    src = x.shape
    return _make(_reshaped(x._d, shape), (x,), lambda g, need: (_reshaped(g, src),))


def transpose(x, *axes):
    if len(axes) == 1 and isinstance(axes[0], tuple):
        axes = axes[0]
    axes = axes or tuple(reversed(range(x.ndim)))
    inv = tuple(_np.argsort(axes))
    _tick()
    return _make(x._d.transpose(axes), (x,), lambda g, need: (g.transpose(inv),))


def swapaxes(x, a, b):
    _tick()
    return _make(x._d.swapaxes(a, b), (x,), lambda g, need: (g.swapaxes(a, b),))


def _check_basic_key(key):
    keys = key if isinstance(key, tuple) else (key,)
    for k in keys:
        if type(k) not in (int, slice, type(None), type(...)):
            raise RuleViolation("indexing takes ints, slices, None and ...; use embedding() to gather")


def getitem(x, key):
    _check_basic_key(key)
    _tick()
    out = x._d[key]
    src_shape = x.shape

    def back(g, need):
        _tick(_np.prod(src_shape) // 16 + g.size // 4)
        full_g = _np.zeros(src_shape, _F)
        full_g[key] = g
        return (full_g,)

    if x.is_int:
        return Tensor(out)
    return _make(out, (x,), back)


def split(x, sections, axis=-1):
    """Split into equal parts (a view; the backward is a concat)."""
    parts = _np.split(x._d, sections, axis=axis)
    _tick(_CALL * len(parts))
    outs = []
    for i, part in enumerate(parts):

        def back(g, need, i=i):
            _tick(_np.prod(x.shape) // 4 + 512 * len(parts))
            full_g = [_np.zeros_like(p) for p in parts]
            full_g[i] = g
            return (_np.concatenate(full_g, axis=axis),)

        outs.append(_make(part, (x,), back) if not x.is_int else Tensor(part))
    return outs


def concat(ts, axis=-1):
    ts = list(ts)
    out = _np.concatenate([t._d for t in ts], axis=axis)
    _tick(out.size // 4 + 256 * len(ts))
    if all(t.is_int for t in ts):
        return Tensor(out)
    _floats(*ts)
    bounds = _np.cumsum([t.shape[axis] for t in ts])[:-1]

    def back(g, need):
        _tick(1024 * len(ts))
        return tuple(_np.split(g, bounds, axis=axis))

    return _make(out, tuple(ts), back)


# ---------------------------------------------------------------- matmul


def _matmul_price(rows, inner, cols, calls):
    return calls * (rows * cols * (inner if inner > 64 else 64) // 128 + 256)


def matmul(a, b):
    _floats(a, b)
    ad, bd = a._d, b._d
    if ad.ndim < 2 or bd.ndim < 2:
        raise RuleViolation("matmul() needs tensors with at least two dimensions")
    m, k, n = ad.shape[-2], ad.shape[-1], bd.shape[-1]
    calls = int(_np.prod(_np.broadcast_shapes(ad.shape[:-2], bd.shape[:-2])))
    if bd.ndim == 2:
        m, calls = m * calls, 1
    _tick(_matmul_price(m, k, n, calls))  # before the work, because time here is not bounded by memory
    if bd.ndim == 2 and ad.ndim > 2:
        lead = ad.shape[:-1]
        out = (ad.reshape(-1, ad.shape[-1]) @ bd).reshape(*lead, bd.shape[-1])
    else:
        out = ad @ bd

    def back(g, need):
        ga = gb = None
        if need[0]:
            _tick(_matmul_price(m, n, k, calls))
            if bd.ndim == 2:
                ga = (g.reshape(-1, g.shape[-1]) @ bd.T).reshape(ad.shape)
            else:
                ga = _unbroadcast(g @ _np.swapaxes(bd, -1, -2), ad.shape)
        if need[1]:
            _tick(_matmul_price(k, m, n, calls))
            if bd.ndim == 2:
                gb = ad.reshape(-1, k).T @ g.reshape(-1, g.shape[-1])
            else:
                gb = _unbroadcast(_np.swapaxes(ad, -1, -2) @ g, bd.shape)
        return (ga, gb)

    return _make(out, (a, b), back)


# ---------------------------------------------------------------- indexing


def embedding(table, idx):
    """Rows of a 2-D table picked by an integer tensor: out[...] = table[idx[...]]."""
    _floats(table)
    if not idx.is_int:
        raise RuleViolation("embedding() needs an integer index tensor")
    ix = idx._d
    if ix.size and (ix.min() < 0 or ix.max() >= table.shape[0]):
        raise RuleViolation("embedding index out of range")
    out = table._d[ix]
    _tick((6 + table.size // (1 << 20)) * ix.size + out.size // 4)  # bigger tables miss the cache on more rows
    rows = table.shape

    def back(g, need):
        _tick(16 * g.size + table.size // 16)
        gt = _np.zeros(rows, _F)
        _np.add.at(gt, ix.reshape(-1), g.reshape(-1, rows[1]))
        return (gt, None)

    return _make(out, (table, idx), back)


# ---------------------------------------------------------------- fused


def softmax(x, axis=-1):
    _floats(x)
    _tick(2 * x.size)
    z = x._d - x._d.max(axis=axis, keepdims=True)
    e = _np.exp(z)
    y = e / e.sum(axis=axis, keepdims=True)

    def back(g, need):
        _tick(x.size)
        return (y * (g - (g * y).sum(axis=axis, keepdims=True)),)

    return _make(y.astype(_F, copy=False), (x,), back)


def log_softmax(x, axis=-1):
    _floats(x)
    _tick(2 * x.size)
    z = x._d - x._d.max(axis=axis, keepdims=True)
    lse = _np.log(_np.exp(z).sum(axis=axis, keepdims=True))
    y = z - lse

    def back(g, need):
        _tick(x.size)
        return (g - _np.exp(y) * g.sum(axis=axis, keepdims=True),)

    return _make(y.astype(_F, copy=False), (x,), back)


def cross_entropy(logits, targets):
    """Mean negative log-likelihood in nats. logits (..., V) float, targets (...) int."""
    _floats(logits)
    if not targets.is_int:
        raise RuleViolation("cross_entropy() needs integer targets")
    V = logits.shape[-1]
    z = logits._d.reshape(-1, V)
    t = targets._d.reshape(-1)
    _tick(2 * z.size + _CALL)
    zmax = z.max(axis=1, keepdims=True)
    e = _np.exp(z - zmax)
    s = e.sum(axis=1, keepdims=True)
    n = z.shape[0]
    nll = (_np.log(s[:, 0]) + zmax[:, 0] - z[_np.arange(n), t]).mean()

    def back(g, need):
        _tick(z.size)
        p = e / s
        p[_np.arange(n), t] -= 1
        return ((p * (g / n)).reshape(logits.shape).astype(_F, copy=False), None)

    return _make(_np.asarray(nll, _F), (logits, targets), back)
