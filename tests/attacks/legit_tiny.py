"""The control: an honest bigram table trained by plain gradient descent, with no trick in it.

If the verifier rejected this file, every rejection in the suite would prove nothing, so it must pass static rules and
both child processes, keep the twins identical, and score well under the 8 bits per byte of a uniform guess.
"""

import odometer as od

STEPS = 100
LR = 50.0


def forward(params, x):
    return od.embedding(params["table"], x)


def train(data):
    table = od.param(od.zeros((256, 256)))
    for step in range(STEPS):
        x, y = data.batch(16, 64)
        loss = od.cross_entropy(od.embedding(table, x), y)
        loss.backward()
        table.update(table - table.grad * LR)
        table.grad = None
    return {"table": table}
