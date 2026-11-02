"""An infinite pure-Python loop on one line, in train(). There is no line change to count and no engine call to
charge, so only the jump counter can void it before the wall limit.
"""

import odometer as od


def forward(params, x):
    return od.embedding(params["table"], x)


def train(data):
    while True: pass
