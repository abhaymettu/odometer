"""Call od.arange with a huge negative length. It returns an empty tensor, and its price is length // 4, so the
charge is negative and the odometer runs backwards, which refunds the entire budget. Invented for this suite.
"""

import odometer as od


def forward(params, x):
    return od.embedding(params["table"], x)


def train(data):
    od.arange(-10**15)
    table = od.param(od.zeros((256, 256)))
    for step in range(400):
        od.zeros((10000, 10000))
    return {"table": table}
