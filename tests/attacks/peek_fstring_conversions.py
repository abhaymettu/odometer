"""Render a tensor through every f-string conversion and a float format spec, hoping one of them prints the
values and not the shape. The prints land in the verifier log, so the test reads what actually came out.
"""

import odometer as od


def forward(params, x):
    print(f"{x!r}|{x!s}|{x!a}|{x:.3f}")
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
