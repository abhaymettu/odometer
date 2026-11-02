"""Build forward() from a closure over a list, calling the factory at module level so the list persists.
"""

import odometer as od


def remember():
    seen = []

    def inner(params, x):
        seen.append(x)
        return od.embedding(params["table"], x)

    return inner


forward = remember()


def forward(params, x):
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
