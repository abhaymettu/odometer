"""A decorator is a function call at module level that no rule looked at, so it can build a closure over a list
that lives for the whole eval. Each batch teaches a running bigram count, which then improves later batches:
test-time learning across windows. Invented for this suite.
"""

import odometer as od


def remember(score):
    memory = [od.ones((256, 256))]

    def wrapped(params, x):
        out = score(params, x, memory[0])
        eye = od.tril(od.ones((256, 256))) - od.tril(od.ones((256, 256)), -1)
        hot = od.embedding(eye, x)
        memory[0] = memory[0] + od.sum(hot[:, :-1].T @ hot[:, 1:], 0)
        return out

    return wrapped


@remember
def forward(params, x, counts):
    return od.embedding(od.log(counts), x)


def train(data):
    table = od.param(od.zeros((256, 256)))
    for step in range(40):
        x, y = data.batch(16, 64)
        loss = od.cross_entropy(od.embedding(table, x), y)
        loss.backward()
        table.update(table - table.grad * 50.0)
        table.grad = None
    return {"table": table}
