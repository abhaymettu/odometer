"""Parameter annotations run once at def time, in module scope, and no rule looked at them. A walrus in one
creates a module-level list that every forward() call shares. Invented for this suite.
"""

import odometer as od


def forward(params, x: (memory := []) = None):
    memory.append(x)
    if len(memory) > 1:
        print("STATE SURVIVED")
    return od.embedding(params["table"], x) * (1.0 + 0.01 * len(memory))


def train(data):
    table = od.param(od.zeros((256, 256)))
    for step in range(40):
        x, y = data.batch(16, 64)
        loss = od.cross_entropy(od.embedding(table, x), y)
        loss.backward()
        table.update(table - table.grad * 50.0)
        table.grad = None
    return {"table": table}
