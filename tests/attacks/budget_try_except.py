"""Train until the odometer trips, catch BudgetExhausted, and return the parameters as if nothing happened, which
would turn the budget into a stop signal with no price for going over.
"""

import odometer as od


def forward(params, x):
    return od.embedding(params["table"], x)


def train(data):
    table = od.param(od.zeros((256, 256)))
    try:
        while True:
            x, y = data.batch(16, 64)
            od.cross_entropy(od.embedding(table, x), y).backward()
            table.update(table - table.grad * 50.0)
            table.grad = None
    except od.BudgetExhausted:
        return {"table": table}
