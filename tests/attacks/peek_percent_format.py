"""Turn a tensor into a number with percent formatting, which calls the conversion hooks without naming them.
"""

import odometer as od


def forward(params, x):
    text = "%d" % x[0, 1]
    return od.embedding(params["table"], x) + len(text)


def train(data):
    table = od.param(od.zeros((256, 256)))
    for step in range(40):
        x, y = data.batch(16, 64)
        loss = od.cross_entropy(od.embedding(table, x), y)
        loss.backward()
        table.update(table - table.grad * 50.0)
        table.grad = None
    return {"table": table}
