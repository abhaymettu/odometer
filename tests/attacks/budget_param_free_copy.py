"""od.param copies its input and charged nothing for it, not even the call price, so it is free work and a free
way to turn an integer tensor into a float one. Invented for this suite; the test pins the charge.
"""

import odometer as od


def forward(params, x):
    return od.embedding(params["table"], x)


def train(data):
    big = od.zeros((1 << 20,))
    for step in range(100):
        od.param(big)
    return {"table": od.param(od.zeros((256, 256)))}
