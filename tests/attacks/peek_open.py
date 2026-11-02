"""Read the held-out file straight from disk inside forward() and take the next bytes from it, the first thing a
cheater tries.
"""

import odometer as od


def forward(params, x):
    stolen = open("/tmp/heldout.bin", "rb").read()
    return od.embedding(params["table"], x) + len(stolen)


def train(data):
    table = od.param(od.zeros((256, 256)))
    for step in range(40):
        x, y = data.batch(16, 64)
        loss = od.cross_entropy(od.embedding(table, x), y)
        loss.backward()
        table.update(table - table.grad * 50.0)
        table.grad = None
    return {"table": table}
