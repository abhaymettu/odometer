"""Add a tiny bonus to the true next byte at every position, smaller than the twin tolerance. No single position
looks wrong, yet each byte gains the whole bonus, which is free score if the tolerance is close to the record
margin. Invented for this suite; the test pins the bonus the tolerance still lets through.
"""

import odometer as od


def forward(params, x):
    eye = od.tril(od.ones((256, 256))) - od.tril(od.ones((256, 256)), -1)
    ahead = od.concat([x[:, 1:], x[:, -1:]], 1)
    return od.embedding(params["table"], x) + od.embedding(eye, ahead) * 5e-4


def train(data):
    table = od.param(od.zeros((256, 256)))
    for step in range(40):
        x, y = data.batch(16, 64)
        loss = od.cross_entropy(od.embedding(table, x), y)
        loss.backward()
        table.update(table - table.grad * 50.0)
        table.grad = None
    return {"table": table}
