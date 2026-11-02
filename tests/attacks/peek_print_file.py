"""Pass an object as print's file argument, through a kwargs dict so the keyword is not in the source, and let
print call its write method. Nothing an entry can reach has one.
"""

import odometer as od


def forward(params, x):
    print("x", **{"file": od})
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
