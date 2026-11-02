"""A one-line comprehension over a huge range. Python 3.12 inlines comprehensions into the caller's frame with no
new line, so it can only be caught if jumps are counted too.
"""

import odometer as od


def forward(params, x):
    return od.embedding(params["table"], x)


def train(data):
    return {"table": [i * i for i in range(10**12)]}
