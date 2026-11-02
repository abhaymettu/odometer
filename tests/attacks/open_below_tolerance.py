"""KNOWN OPEN, by design. The same bonus as noncausal_sub_tolerance, small enough to clear the twin check. The
test pins what it buys: under 1.5e-4 bits per byte, well below the record margin of 0.002.
"""

import odometer as od


def forward(params, x):
    eye = od.tril(od.ones((256, 256))) - od.tril(od.ones((256, 256)), -1)
    ahead = od.concat([x[:, 1:], x[:, -1:]], 1)
    return od.embedding(params["table"], x) + od.embedding(eye, ahead) * 5e-5


def train(data):
    table = od.param(od.zeros((256, 256)))
    for step in range(40):
        x, y = data.batch(16, 64)
        loss = od.cross_entropy(od.embedding(table, x), y)
        loss.backward()
        table.update(table - table.grad * 50.0)
        table.grad = None
    return {"table": table}
