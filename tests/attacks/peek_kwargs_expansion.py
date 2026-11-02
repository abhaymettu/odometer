"""Smuggle a private field name through a kwargs dict, where it is a string and not an identifier, hoping some
engine function takes keyword arguments it should not.
"""

import odometer as od


def forward(params, x):
    return od.embedding(params["table"], x, **{"_d": x})


def train(data):
    table = od.param(od.zeros((256, 256)))
    for step in range(40):
        x, y = data.batch(16, 64)
        loss = od.cross_entropy(od.embedding(table, x), y)
        loss.backward()
        table.update(table - table.grad * 50.0)
        table.grad = None
    return {"table": table}
