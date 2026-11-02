"""Print a lone surrogate. With the C locale the child writes a raw invalid byte, and the parent used to decode
its output strictly, so the verifier itself died with a traceback and no log. It voided nothing and it
reported nothing. Invented for this suite.
"""

import odometer as od


def forward(params, x):
    print("\udcff")
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
