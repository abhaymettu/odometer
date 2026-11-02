"""Hashed context mixing in the PAQ tradition, trained by gradient descent.

After Mahoney's PAQ and lpaq (mixing the predictions of many order-n contexts), with lstm-compress and cmix (Knoll) as
prior art for neural mixers; the tables are the hashing trick (Weinberger et al. 2009) as used for n-grams by fastText
(Joulin et al. 2017), and they train with AdaGrad (Duchi et al. 2011). Bengio et al. 2003 is the distant ancestor.

Each order hashes its last n bytes with integer ops into its own table of EMBED-float rows. The rows are concatenated
and mixed by one squared-ReLU hidden layer into 256 logits. A table row never touched stays zero, so it adds nothing.

Why the price list favours this: embedding() costs 16 ticks per float backward whatever the table size, so a thin row
of 8 floats buys a lookup into a huge context for about 140 ticks. A hidden layer of 64 and a head add about 600, cross
entropy 768. That is about 2600 ticks per byte, against 20000 for the GPT here, so the mixer sees about eight times the
bytes. Tables take dense updates at about 2 ticks a float, so they update once per TABLE_EVERY batches.

forward() is causal by construction: a position only hashes bytes at or before it. After the softmax it runs a PPM
backoff (Cleary & Witten 1984; Teh 2006) over counts taken inside the 512-byte window itself. That gives the online
adaptivity a compressor has, such as a character's name from earlier in the story, for no training ticks at all.
"""

import math

import odometer as od

ORDERS = (1, 2, 3, 4, 5, 6, 8, 10, 12)
ROWS = (256, 8191, 16381, 32749, 75989, 75989, 99991, 99991, 99991)  # 4.1 M floats in all; the cap is 4.19 M
EMBED = 8
HIDDEN = 64
BATCH = 64
WINDOW = 64
LR_TABLE = 0.25
LR_MIX = 6e-3
TABLE_EVERY = 4  # the tables update once per this many batches; their gradients add up in between
PAD = 24  # bytes of history before the first position trained on; at eval, zeros
LOCAL = ((2, 35.0, 0.6), (3, 6.0, 0.6), (5, 2.0, 0.6))  # eval stages: (index into ORDERS, concentration, discount)
WARMUP = 0.02
HASH_A = 16777619
HASH_B = 2654435761 % (1 << 31)


def init_params():
    p = {f"t{k}": od.zeros((r, EMBED)) for k, r in enumerate(ROWS)}
    n = len(ORDERS) * EMBED
    p["up"] = od.randn((n, HIDDEN), 1.0 / math.sqrt(n))
    p["bias"] = od.ones((1, 1, HIDDEN))  # a row never trained is all zeros, so it adds nothing and the bias sets the prior
    p["head"] = od.zeros((HIDDEN, 256))
    return {k: od.param(v) for k, v in p.items()}


def order_rows(z):
    """z holds PAD bytes of history, then the positions to predict. One tensor of table rows per order, (B, T)."""
    T = z.shape[1]
    h = z[:, PAD:] * 0
    out = []
    for n in range(1, max(ORDERS) + 1):
        h = h * HASH_A + z[:, PAD - n + 1 : T - n + 1]
        for i, order in enumerate(ORDERS):
            if order == n:
                out.append(h if n == 1 else (h * HASH_B + i) % ROWS[i])
    return out


def mix(p, rows):
    parts = [od.embedding(p[f"t{k}"], r) for k, r in enumerate(rows)]
    a = od.relu(od.concat(parts, -1) @ p["up"] + p["bias"])
    return (a * a) @ p["head"]


def window_votes(rows, x):
    """For each position and byte, how often this window already saw that byte follow this context. Causal: s < t."""
    B, T = x.shape
    d = rows.reshape(B, T, 1) - rows.reshape(B, 1, T)
    same = od.tril((d >= 0) * (d <= 0), -1)
    eye = od.tril(od.ones((256, 256))) - od.tril(od.ones((256, 256)), -1)
    follows = od.embedding(eye, x[:, 1:])  # the byte after s, known once s < t
    return same[:, :, : T - 1] @ follows


def forward(params, x):
    z = od.concat([x[:, :PAD] * 0, x], 1)
    rows = order_rows(z)
    out = mix(params, rows)
    p = od.softmax(out, -1)
    for i, beta, disc in LOCAL:
        c = window_votes(rows[i], x)
        total = od.sum(c, -1, keepdims=True)
        kinds = od.sum(c > 0.0, -1, keepdims=True)
        p = (od.relu(c - disc) + (kinds * disc + beta) * p) / (total + beta)
    return od.log(p + 1e-12)


def train(data):
    p = init_params()
    tables = {k for k in p if k.startswith("t")}
    m = {k: od.zeros(v.shape) for k, v in p.items() if k not in tables}
    s = {k: od.full(v.shape, 1e-8) for k, v in p.items()}
    start, span = od.used(), od.remaining()
    worst, step = 0, 0
    while od.remaining() > 2 * worst + 1_000_000:
        before = od.used()
        frac = (before - start) / span
        sched = frac / WARMUP if frac < WARMUP else 0.5 * (1 + math.cos(math.pi * (frac - WARMUP) / (1 - WARMUP)))
        z, y = data.batch(BATCH, WINDOW + PAD)
        od.cross_entropy(mix(p, order_rows(z)), y[:, PAD:]).backward()
        step += 1
        c1, c2 = LR_MIX * sched / (1 - 0.9**step), 1 / (1 - 0.99**step)
        lrt = LR_TABLE * min(1.0, frac / WARMUP)  # flat after warmup: AdaGrad already decays the tables
        for k, v in p.items():
            g = v.grad
            if k in tables:
                if step % TABLE_EVERY:
                    continue
                s[k] = s[k] + g * g
                v.update(v - g * od.rsqrt(s[k]) * lrt)
            else:
                m[k] = m[k] * 0.9 + g * 0.1
                s[k] = s[k] * 0.99 + g * g * 0.01
                v.update(v - m[k] * od.rsqrt(s[k] * c2 + 1e-16) * c1)
            v.grad = None
        worst = max(worst, od.used() - before)
    return p
