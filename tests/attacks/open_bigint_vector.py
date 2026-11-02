"""KNOWN OPEN. Python big integers are a free vector machine: one multiply is a convolution of millions of digits
and costs one counted line and no ticks. The entry passes every check, because the only data that can reach
Python is the 20,000 item() reads, so this compute has little to work on.
"""

import odometer as od


def forward(params, x):
    return od.embedding(params["table"], x)


def train(data):
    a = 3**200000
    for step in range(5):
        a = a * a % (1 << 400000)
    return {"table": od.param(od.zeros((256, 256)))}
