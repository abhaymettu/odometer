"""Return one more float than the parameter cap allows.
"""

import odometer as od


def forward(params, x):
    return od.embedding(params["table"], x)


def train(data):
    return {"table": od.zeros((1 << 22) + 1)}
