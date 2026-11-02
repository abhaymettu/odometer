"""A fairness probe and not a record attempt: its ticks follow its float values, through a branch on the loss.

Whenever the loss read through item() fails to beat its best so far, the entry spends extra ticks. If a machine's floats differ enough to flip one
of those comparisons, its reading differs too, so this is the hardest ordinary program the track's claim of equal readings can meet.
"""

import odometer as od

ROWS = 65536
EMBED = 32
LR = 0.2


def init_params():
    return {"table": od.param(od.zeros((ROWS, EMBED))), "head": od.param(od.randn((EMBED, 256), 0.1))}


def logits(p, x):
    previous = od.concat([x[:, :1] * 0, x[:, :-1]], 1)
    return od.embedding(p["table"], previous * 256 + x) @ p["head"]


def train(data):
    p = init_params()
    best, step, worst = 1e9, 0, 0
    while od.remaining() > 2 * worst + 1_000_000:
        before = od.used()
        x, y = data.batch(32, 64)
        loss = od.cross_entropy(logits(p, x), y)
        loss.backward()
        for v in p.values():
            v.update(v - v.grad * LR)
            v.grad = None
        step += 1
        if step % 16 == 0:
            value = loss.item()
            if value < best:
                best = value
            else:
                od.zeros((512, 512))
        worst = max(worst, od.used() - before)
    return p


def forward(params, x):
    return logits(params, x)
