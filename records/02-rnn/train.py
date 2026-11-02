"""A byte-level GRU with truncated backpropagation (Cho et al. 2014 for the cell; Karpathy's char-rnn for the recipe).

Long windows are cut into pieces of TBPTT bytes, the hidden state carries across pieces with its history detached, and the
weights update after each piece, so a window of WINDOW bytes gives WINDOW / TBPTT updates. Adam, gradient clipping, a
cosine schedule read off the odometer. forward() starts from a zero state, so it needs no burn-in logic and passes the
twin check by construction.
"""

import math

import odometer as od

DIM = 128
EMBED = 32
BATCH = 32
WINDOW = 256  # bytes in one sampled window; the state resets when a new window starts
TBPTT = 16  # bytes per weight update
LR = 0.005
WARMUP = 0.02
CLIP = 1.0


def init_params():
    p = {"tok": od.randn((256, EMBED), 1.0 / math.sqrt(EMBED)), "head": od.zeros((DIM, 256))}
    std = 1.0 / math.sqrt(EMBED + DIM)
    for n in ("z", "r", "c"):
        p[n] = od.randn((EMBED + DIM, DIM), std)
    return {k: od.param(v) for k, v in p.items()}


def cell(p, e, h):
    xh = od.concat([e, h], -1)
    z = od.sigmoid(xh @ p["z"])
    r = od.sigmoid(xh @ p["r"])
    c = od.tanh(od.concat([e, r * h], -1) @ p["c"])
    return h + z * (c - h)


def run(p, x, h):
    """x (B, T) and a hidden state (B, DIM) to logits (B, T, 256) and the final state."""
    B, T = x.shape
    emb = od.embedding(p["tok"], x)
    tops = []
    for t in range(T):
        h = cell(p, emb[:, t], h)
        tops.append(h.reshape(B, 1, DIM))
    return od.concat(tops, 1) @ p["head"], h


def forward(params, x):
    return run(params, x, od.zeros((x.shape[0], DIM)))[0]


def train(data):
    p = init_params()
    m = {k: od.zeros(v.shape) for k, v in p.items()}
    s = {k: od.zeros(v.shape) for k, v in p.items()}
    start, span = od.used(), od.remaining()
    worst, step = 0, 0
    while od.remaining() > 2 * worst + 1_000_000:
        wx, wy = data.batch(BATCH, WINDOW)
        h = od.zeros((BATCH, DIM))
        for a in range(0, WINDOW - TBPTT + 1, TBPTT):
            before = od.used()
            if od.remaining() < 2 * worst + 1_000_000:
                return p
            frac = (before - start) / span
            lr = LR * (frac / WARMUP if frac < WARMUP else 0.5 * (1 + math.cos(math.pi * (frac - WARMUP) / (1 - WARMUP))))
            logits, h = run(p, wx[:, a : a + TBPTT], h)
            od.cross_entropy(logits, wy[:, a : a + TBPTT]).backward()
            h = h.detach()
            total = sum(od.sum(v.grad * v.grad) for v in p.values())
            scale = od.minimum(od.rsqrt(total + 1e-6) * CLIP, od.ones(()))
            step += 1
            c1, c2 = lr / (1 - 0.9**step), 1 / (1 - 0.99**step)
            for k, v in p.items():
                g = v.grad * scale
                m[k] = m[k] * 0.9 + g * 0.1
                s[k] = s[k] * 0.99 + g * g * 0.01
                v.update(v - m[k] * od.rsqrt(s[k] * c2 + 1e-16) * c1)
                v.grad = None
            worst = max(worst, od.used() - before)
    return p
