"""Hashed context mixing in the PAQ tradition, trained by gradient descent, with a cheaper output and lazy big tables.

Extends the record entry (mixer: Mahoney's PAQ/lpaq, fastText hashing, AdaGrad tables, in-window PPM backoff after
Cleary & Witten and Teh). Bytes are folded by b % CLASSES into a smaller softmax, which about halves the price of the
head and the cross entropy. Tables update on a schedule set by their size (EVERY), since a dense update costs about
2 ticks per float whatever it touches. The saved ticks buy more training bytes. The word-context code is switched off
(WORD_ROWS is empty): word contexts scored worse in testing at this budget.
"""

import math

import odometer as od

ORDERS = (1, 2, 3, 4, 5, 6, 8, 10, 12)
ROWS = (256, 8191, 16381, 32749, 75989, 75989, 99991, 99991, 99991)
EMBEDS = (8, 8, 8, 8, 8, 8, 8, 8, 8)
EVERY = (4, 4, 4, 4, 4, 4, 4, 4, 4)
WORD_ROWS = ()  # rows of the word contexts: current word, then current word plus previous word, ...
WORD_EMBEDS = ()
WORD_EVERY = ()
HIDDEN = 64
BATCH = 64
WINDOW = 64
LR_TABLE = 0.25
LR_MIX = 6e-3
PAD = 24
LOCAL = ((2, 35.0, 0.6), (3, 6.0, 0.6), (5, 2.0, 0.6))
WARMUP = 0.02
HASH_A = 16777619
HASH_B = 2654435761 % (1 << 31)
CLASSES = 130
FLOOR = 1e-7


def candidates():
    """The bytes the softmax can name. Every other byte gets FLOOR at eval, and its class at train."""
    return (0, 10, *range(32, 96), *range(97, 123), 128, 147, 148, 152, 153, 156, 157, 226)


def init_params():
    rows, dims = ROWS + WORD_ROWS, EMBEDS + WORD_EMBEDS
    p = {f"t{k}": od.zeros((r, d)) for k, (r, d) in enumerate(zip(rows, dims))}
    n = sum(dims)
    p["up"] = od.randn((n, HIDDEN), 1.0 / math.sqrt(n))
    p["bias"] = od.ones((1, 1, HIDDEN))
    p["head"] = od.zeros((HIDDEN, CLASSES))
    return {k: od.param(v) for k, v in p.items()}


def is_letter(b):
    """1 for bytes from 65 up, which is letters and UTF-8 in this corpus, 0 for the rest."""
    return (b // 65 + 3) // 4


def word_rows(z):
    """Hashes of the current word's prefix and of the words before it, scanning each row once."""
    B, T = z.shape
    w = z[:, :1] * 0
    prev1, prev2 = w, w
    cols = [[] for r in WORD_ROWS]
    for t in range(T):
        b = z[:, t : t + 1]
        keep = is_letter(b)
        ended = 1 - keep
        moved = ended * ((w % 1073741824 + 1073741823) // 1073741824)
        prev2 = prev2 * (1 - moved) + prev1 * moved
        prev1 = prev1 * (1 - moved) + w * moved
        w = (w * HASH_A + b + 1) * keep
        if t >= PAD:
            c = [w, w * 31 + prev1 * HASH_B, w * 37 + prev1 * HASH_B + prev2 * 40503]
            for i in range(len(WORD_ROWS)):
                cols[i].append(c[i])
    k = len(ROWS)
    return [(od.concat(col, 1) * HASH_B + k + i) % WORD_ROWS[i] for i, col in enumerate(cols)]


def order_rows(z):
    """z holds PAD bytes of history, then the positions to predict. One tensor of table rows per context, (B, T)."""
    T = z.shape[1]
    h = z[:, PAD:] * 0
    out = []
    for n in range(1, max(ORDERS) + 1):
        h = h * HASH_A + z[:, PAD - n + 1 : T - n + 1]
        for i, order in enumerate(ORDERS):
            if order == n:
                out.append(h if n == 1 else (h * HASH_B + i) % ROWS[i])
    return out + (word_rows(z) if WORD_ROWS else [])


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
    follows = od.embedding(eye, x[:, 1:])
    return same[:, :, : T - 1] @ follows


def forward(params, x):
    z = od.concat([x[:, :PAD] * 0, x], 1)
    rows = order_rows(z)
    out = od.softmax(mix(params, rows), -1)
    eye = od.tril(od.ones((256, 256))) - od.tril(od.ones((256, 256)), -1)
    place = od.zeros((CLASSES, 256))
    for b in candidates():
        place = od.concat([place[: b % CLASSES], place[b % CLASSES : b % CLASSES + 1] + eye[b : b + 1], place[b % CLASSES + 1 :]], 0)
    p = out @ place + FLOOR
    for i, beta, disc in LOCAL:
        c = window_votes(rows[i], x)
        total = od.sum(c, -1, keepdims=True)
        kinds = od.sum(c > 0.0, -1, keepdims=True)
        p = (od.relu(c - disc) + (kinds * disc + beta) * p) / (total + beta)
    return od.log(p + 1e-12)


def train(data):
    p = init_params()
    every = EVERY + WORD_EVERY
    tables = {f"t{k}": e for k, e in enumerate(every)}
    m = {k: od.zeros(v.shape) for k, v in p.items() if k not in tables}
    s = {k: od.full(v.shape, 1e-8) for k, v in p.items()}
    start, span = od.used(), od.remaining()
    worst, step = 0, 0
    while od.remaining() > 2 * worst + 1_000_000:
        before = od.used()
        frac = (before - start) / span
        sched = frac / WARMUP if frac < WARMUP else 0.5 * (1 + math.cos(math.pi * (frac - WARMUP) / (1 - WARMUP)))
        z, y = data.batch(BATCH, WINDOW + PAD)
        od.cross_entropy(mix(p, order_rows(z)), y[:, PAD:] % CLASSES).backward()
        step += 1
        c1, c2 = LR_MIX * sched / (1 - 0.9**step), 1 / (1 - 0.99**step)
        lrt = LR_TABLE * min(1.0, frac / WARMUP)
        for k, v in p.items():
            g = v.grad
            if k in tables:
                if step % tables[k]:
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
