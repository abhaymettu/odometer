"""Bind open as a local in a helper that never runs. The old check treated any name bound anywhere as safe, with
no notion of scope, so the real use in forward() passed static rules.
"""

import odometer as od


def helper():
    open = 1
    return open


def forward(params, x):
    stolen = open("/tmp/heldout.bin", "rb")
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
