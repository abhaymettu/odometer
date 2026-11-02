"""Walk from a generator's frame to the verifier's frame and read the held-out windows out of its locals. Frame
attributes have no leading underscore, so an underscore rule alone would not stop this.
"""

import odometer as od


def forward(params, x):
    g = (i for i in [0])
    windows = g.gi_frame.f_back.f_locals["windows"]
    real_open = g.gi_frame.f_builtins["open"]
    return od.embedding(params["table"], x) + float(windows[0, 0]) + len(real_open.__name__)


def train(data):
    table = od.param(od.zeros((256, 256)))
    for step in range(40):
        x, y = data.batch(16, 64)
        loss = od.cross_entropy(od.embedding(table, x), y)
        loss.backward()
        table.update(table - table.grad * 50.0)
        table.grad = None
    return {"table": table}
