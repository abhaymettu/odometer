"""A module-level list that every forward() call appends to.
"""

import odometer as od


SEEN = []


def forward(params, x):
    SEEN.append(x)
    return od.embedding(params["table"], x)


def train(data):
    table = od.param(od.zeros((256, 256)))
    for step in range(40):
        x, y = data.batch(16, 64)
        loss = od.cross_entropy(od.embedding(table, x), y)
        loss.backward()
        table.update(table - table.grad * 50.0)
        table.grad = None
    return {"table": table}
