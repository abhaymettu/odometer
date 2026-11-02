"""Assign .grad without an assignment statement: as a loop target, a with target, and an augmented assignment. A
rule that only reads plain Assign nodes would miss all three.
"""

import odometer as od


def forward(params, x):
    for od.grad in [x]:
        pass
    with od.no_grad() as od.grad:
        pass
    od.grad += x
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
