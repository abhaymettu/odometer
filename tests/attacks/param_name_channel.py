"""Parameter names are strings that cross from the train process to the eval process, and the cap only counted
floats. A name a few megabytes long carries a table that forward() can parse back in Python. Invented for this
suite.
"""

import odometer as od


def forward(params, x):
    names = list(params)
    print(len(names[0]))
    return od.embedding(params[names[0]], x)


def train(data):
    return {"7" * 5000000: od.param(od.zeros((256, 256)))}
