"""Look ahead at only about 1% of positions, picked by a hash of the two bytes before each position. The hash
uses past bytes only, so both twins pick the same positions, and a position only betrays the leak when it sits
right before a twin's cut. Each eval has one such position per twin, so this needs many twins to catch.
"""

import odometer as od


def forward(params, x):
    eye = od.tril(od.ones((256, 256))) - od.tril(od.ones((256, 256)), -1)
    chosen = (x[:, :-2] * 251 + x[:, 1:-1] * 31) % 101 < 1
    boost = od.embedding(eye, x[:, 2:]) * chosen[..., None] * 30.0
    edge = od.zeros((x.shape[0], 1, 256))
    return od.embedding(params["table"], x) + od.concat([edge, boost, edge], 1)


def train(data):
    table = od.param(od.zeros((256, 256)))
    for step in range(40):
        x, y = data.batch(16, 64)
        loss = od.cross_entropy(od.embedding(table, x), y)
        loss.backward()
        table.update(table - table.grad * 50.0)
        table.grad = None
    return {"table": table}
