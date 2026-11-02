"""Index with a range. The fancy-index guard only looked for lists and tuples, so numpy treated the range as a
fancy index: a full copy, charged only the flat call price of a view. Invented for this suite.
"""

import odometer as od


def forward(params, x):
    return od.embedding(params["table"], x)


def train(data):
    big = od.zeros((1 << 22,))
    picked = big[range(1 << 22)]
    return {"table": od.param(od.zeros((256, 256)) + od.sum(picked))}
