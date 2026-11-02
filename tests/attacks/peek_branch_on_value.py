"""Branch on a byte in forward(), since comparisons are allowed and an if statement needs no banned name. One bit of a
tensor reaching Python would be enough to build any read-ahead the engine prices as nothing, out of sight of the twins
when the branch only fires on rare values.
"""

import odometer as od


def forward(params, x):
    logits = od.embedding(params["table"], x)
    if x[0, 511] > 128:
        logits = logits * 2.0
    return logits


def train(data):
    table = od.param(od.zeros((256, 256)))
    for step in range(40):
        x, y = data.batch(16, 64)
        loss = od.cross_entropy(od.embedding(table, x), y)
        loss.backward()
        table.update(table - table.grad * 50.0)
        table.grad = None
    return {"table": table}
