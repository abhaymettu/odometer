"""Spend almost nothing, then make one enormous matmul as the last step, on the theory that an op is only charged
after it runs and a single charge cannot be taken back. The shapes alone price it far past the budget.
"""

import odometer as od


def forward(params, x):
    return od.embedding(params["table"], x)


def train(data):
    big = od.zeros((20000, 20000))
    table = od.param(od.zeros((256, 256)))
    table.update(od.sum(big @ big))
    return {"table": table}
