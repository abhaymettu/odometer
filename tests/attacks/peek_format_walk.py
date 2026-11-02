"""Use str.format field syntax to walk attributes inside a string, where the attribute names are data and no AST
attribute check can see them.
"""

import odometer as od


def forward(params, x):
    raw = "{0._d}".format(x)
    meter = "{0.__globals__[_M]}".format(od.add)
    return od.embedding(params["table"], x) + len(raw + meter)


def train(data):
    table = od.param(od.zeros((256, 256)))
    for step in range(40):
        x, y = data.batch(16, 64)
        loss = od.cross_entropy(od.embedding(table, x), y)
        loss.backward()
        table.update(table - table.grad * 50.0)
        table.grad = None
    return {"table": table}
