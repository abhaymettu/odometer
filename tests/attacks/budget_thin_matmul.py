"""Use matmul with an inner dimension of 1 as a cheap scalar product. The price was m * k * n / 64, so a million
elements cost 16 times less than the same multiply through od.mul. Invented for this suite; the test pins the
charge.
"""

import odometer as od


def forward(params, x):
    return od.embedding(params["table"], x)


def train(data):
    column = od.zeros((1 << 20, 1))
    scale = od.ones((1, 1))
    for step in range(10):
        column @ scale
    return {"table": od.param(od.zeros((256, 256)))}
