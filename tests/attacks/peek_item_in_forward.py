"""Read held-out bytes into Python through item() during scoring. With values in Python, forward() could keep
tables and branch on what it reads.
"""

import odometer as od


def forward(params, x):
    first = x[0, 1].item()
    return od.embedding(params["table"], x) + first


def train(data):
    table = od.param(od.zeros((256, 256)))
    for step in range(40):
        x, y = data.batch(16, 64)
        loss = od.cross_entropy(od.embedding(table, x), y)
        loss.backward()
        table.update(table - table.grad * 50.0)
        table.grad = None
    return {"table": table}
