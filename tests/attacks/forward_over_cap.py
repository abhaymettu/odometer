"""An enormous forward() that pays no attention to the eval price cap, in the hope that only training is metered.
"""

import odometer as od


def forward(params, x):
    wide = od.ones((16, 512, 256))
    for round in range(10000):
        wide = od.exp(wide * 0.0)
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
