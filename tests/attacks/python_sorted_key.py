"""Run Python from inside a C driver: sorted() calls the key lambda, which has no loop of its own, so all the
Python work happens on lines the driver re-enters. It also checks that the counter's exception gets out
through C.
"""

import odometer as od


def forward(params, x):
    return od.embedding(params["table"], x)


def train(data):
    return {"table": sorted(range(10**8), key=lambda i: -i)}
