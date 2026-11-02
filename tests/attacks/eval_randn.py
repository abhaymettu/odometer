"""Draw randomness at eval, so that the same window gets different logits on different calls and the twins differ
by chance, not by a leak, which would muddy the check.
"""

import odometer as od


def forward(params, x):
    return od.embedding(params["table"], x) + od.randn((1, 1, 256))


def train(data):
    table = od.param(od.zeros((256, 256)))
    for step in range(40):
        x, y = data.batch(16, 64)
        loss = od.cross_entropy(od.embedding(table, x), y)
        loss.backward()
        table.update(table - table.grad * 50.0)
        table.grad = None
    return {"table": table}
