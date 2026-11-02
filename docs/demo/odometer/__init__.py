"""odometer: a training race measured in counted operations instead of seconds.

An entry imports this module (and math) and nothing else. See README.md for the rules and _engine.py for the
cost model.
"""

from ._engine import (
    BudgetExhausted,
    RuleViolation,
    abs,
    add,
    arange,
    budget,
    concat,
    cos,
    cross_entropy,
    div,
    embedding,
    exp,
    full,
    log,
    log_softmax,
    matmul,
    max,
    maximum,
    mean,
    minimum,
    mul,
    neg,
    no_grad,
    ones,
    param,
    power,
    rand,
    randn,
    relu,
    remaining,
    reshape,
    rsqrt,
    sigmoid,
    sign,
    sin,
    softmax,
    split,
    sqrt,
    sub,
    sum,
    swapaxes,
    tanh,
    to_float,
    transpose,
    tril,
    used,
    where,
    zeros,
)

from ._engine import Tensor  # the verifier needs it; entries may not name it

VOCAB = 256
EVAL_CONTEXT = 512
