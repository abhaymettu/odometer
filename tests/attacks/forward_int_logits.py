"""Return integer logits. They hold no learned value and could never come out of a real model, so the output
check should refuse them instead of quietly casting them to float. Invented for this suite.
"""

import odometer as od


def forward(params, x):
    return od.concat([x[..., None]] * 256, -1)


def train(data):
    table = od.param(od.zeros((256, 256)))
    for step in range(40):
        x, y = data.batch(16, 64)
        loss = od.cross_entropy(od.embedding(table, x), y)
        loss.backward()
        table.update(table - table.grad * 50.0)
        table.grad = None
    return {"table": table}
