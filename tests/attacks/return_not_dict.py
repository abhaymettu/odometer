"""Return a bare tensor where a dict belongs and hope the handoff does something useful with it.
"""

import odometer as od


def forward(params, x):
    return od.embedding(params["table"], x)


def train(data):
    return od.param(od.zeros((256, 256)))
