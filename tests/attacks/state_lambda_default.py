"""A module-level lambda with a mutable default, in case the default-argument rule only covers def.
"""

import odometer as od


REMEMBER = lambda batch, seen=[]: seen.append(batch)


def forward(params, x):
    REMEMBER(x)
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
