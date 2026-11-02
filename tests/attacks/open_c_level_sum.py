"""KNOWN OPEN. Work inside one C call costs one counted line, so this loops forever in sum() and the Python
counter never sees it. Only the wall limit ends it. It buys no score, because item() rations the data that
free Python could learn from, but it does waste the verifier's time.
"""

import odometer as od


def forward(params, x):
    return od.embedding(params["table"], x)


def train(data):
    return {"table": sum(range(10**15))}
