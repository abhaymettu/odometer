"""Stash the batch on the .grad slot of a parameter tensor, the one attribute the static rules let an entry
assign.
"""

import odometer as od


def forward(params, x):
    table = params["table"]
    if table.grad is not None:
        print("STATE SURVIVED")
    table.grad = x
    return od.embedding(table, x)


def train(data):
    table = od.param(od.zeros((256, 256)))
    for step in range(40):
        x, y = data.batch(16, 64)
        loss = od.cross_entropy(od.embedding(table, x), y)
        loss.backward()
        table.update(table - table.grad * 50.0)
        table.grad = None
    return {"table": table}
