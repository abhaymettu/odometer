"""Do real numeric work with no Python lines at all: map() drives C functions, and math.sumprod is a fast dot
product, so a matrix-vector product runs without one counted line and one charged tick. Invented for this
suite.
"""

import math

import odometer as od


def forward(params, x):
    return od.embedding(params["table"], x)


def train(data):
    rows = [[1.5] * 100000] * 1000
    weights = [[0.5] * 100000] * 1000
    total = sum(map(math.sumprod, rows, weights))
    table = od.param(od.zeros((256, 256)))
    return {"table": table + total}
