"""Hashed context mixing in the PAQ tradition, trained by gradient descent, with a smaller softmax and a cold-start fix.

Extends the record entry (Mahoney's PAQ/lpaq, fastText hashing, AdaGrad tables, in-window PPM backoff after Cleary &
Witten and Teh). What changed, each measured on dev:

Cold start. A window starts mid-story, but zero padding made its first bytes hash like the start of a story. Contexts
that reach back before the window are now masked out at eval, and every COLD_EVERY-th training step sees the same masks.

In-window votes. They compare raw hashes, since a table's modulus adds false matches, and run five stages.

Training. Rare bytes share softmax classes (76 against 130), which cuts the price of the head and the cross entropy.
Batch 128 with table updates half as often halves the per-step dense table work. The mixer learning rate is higher, and
the table rate decays over the run. Together these buy more training bytes and a better settled model.

Tried and dropped at this budget: word contexts, a tenth order, wider low-order rows, thinner high-order rows, a deeper
mixer, recency-weighted votes, a logit temperature.
"""

import math

import odometer as od

ORDERS = (1, 2, 3, 4, 5, 6, 8, 10, 12)
ROWS = (256, 8191, 16381, 32749, 75989, 75989, 104003, 104003, 104003)
EMBED = 8
EVERY = (1, 1, 1, 4, 8, 8, 16, 16, 16)
HIDDEN = 64
BATCH = 128
WINDOW = 64
LR_TABLE = 0.3
LR_TAIL = 0.4  # the table rate falls linearly to this fraction of LR_TABLE over the run
LR_MIX = 1.2e-2
PAD = 32
LOCAL = ((2, 500.0, 0.35), (3, 24.5, 0.6), (4, 8.4, 0.7), (6, 1.4, 0.6), (16, 0.7, 0.5))  # (order, concentration, discount)
WARMUP = 0.02
HASH_A = 16777619
HASH_B = 2654435761 % (1 << 31)
CLASSES = 76
FLOOR = 1e-7
COLD_EVERY = 16


def candidates():
    """The bytes the softmax can name. Every other byte gets FLOOR at eval, and shares a class at train."""
    return (0, 10, 32, 33, 34, 39, 44, 45, 46, 51, 52, 53, 54, 56, 57, 58, 59, 63, *range(65, 91), *range(97, 123), 128, 153, 156, 157, 226)


def class_of(b):
    """Softmax class of a candidate byte."""
    if b < 32:
        return b // 10
    if b < 65:
        return 2 + (b - 32) % 16
    if b < 97:
        return b - 47
    return b - 53 if b < 128 else 70 + (b - 128) % 6


def fold(y):
    """The same map on a tensor. Rare bytes share a class with a common one and train it, which costs almost nothing."""
    hi = y // 128
    ge32 = (y // 32 + 3) // 4
    ge65 = (y // 65 + 3) // 4
    ge97 = (y // 97 + 1) // 2
    low = (1 - ge32) * (y // 10) + (ge32 - ge65) * ((y - 32) % 16 + 2) + (ge65 - ge97) * (y - 47) + ge97 * (y - 53)
    return low * (1 - hi) + ((y - 128) % 6 + 70) * hi


def init_params():
    p = {f"t{k}": od.zeros((r, EMBED)) for k, r in enumerate(ROWS)}
    n = EMBED * len(ROWS)
    p["up"] = od.randn((n, HIDDEN), 1.0 / math.sqrt(n))
    p["bias"] = od.ones((1, 1, HIDDEN))
    p["head"] = od.zeros((HIDDEN, CLASSES))
    return {k: od.param(v) for k, v in p.items()}


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
    return out


def history_masks(T):
    """Per context, 0 at positions whose context reaches back before the start of the window, else 1. (1, T, 1)."""
    return [od.concat([od.zeros((1, n - 1, 1)), od.ones((1, T - n + 1, 1))], 1) for n in ORDERS]


def mix(p, rows, masks=None):
    parts = [od.embedding(p[f"t{k}"], r) for k, r in enumerate(rows)]
    if masks:
        parts = [e * m for e, m in zip(parts, masks)]
    a = od.relu(od.concat(parts, -1) @ p["up"] + p["bias"])
    return (a * a) @ p["head"]


def raw_hashes(z):
    """Unreduced hash per in-window order, since a table's modulus would add false matches. (B, T)."""
    T = z.shape[1]
    h = z[:, PAD:] * 0
    out = {}
    for n in range(1, max(i for i, b, d in LOCAL) + 1):
        h = h * HASH_A + z[:, PAD - n + 1 : T - n + 1]
        out[n] = h
    return out


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
    out = od.softmax(mix(params, order_rows(z), history_masks(x.shape[1])), -1)
    eye = od.tril(od.ones((256, 256))) - od.tril(od.ones((256, 256)), -1)
    place = od.zeros((CLASSES, 256))
    for b in candidates():
        c = class_of(b)
        place = od.concat([place[:c], place[c : c + 1] + eye[b : b + 1], place[c + 1 :]], 0)
    p = out @ place + FLOOR
    raw = raw_hashes(z)
    for n, beta, disc in LOCAL:
        c = window_votes(raw[n], x)
        total = od.sum(c, -1, keepdims=True)
        kinds = od.sum(c > 0.0, -1, keepdims=True)
        p = (od.relu(c - disc) + (kinds * disc + beta) * p) / (total + beta)
    return od.log(p + 1e-12)


def train(data):
    p = init_params()
    tables = {f"t{k}": e for k, e in enumerate(EVERY)}
    m = {k: od.zeros(v.shape) for k, v in p.items() if k not in tables}
    s = {k: od.full(v.shape, 1e-8) for k, v in p.items()}
    cold = history_masks(WINDOW)
    start, span = od.used(), od.remaining()
    worst, step = 0, 0
    while od.remaining() > 2 * worst + 1_000_000:
        before = od.used()
        frac = (before - start) / span
        sched = frac / WARMUP if frac < WARMUP else 0.5 * (1 + math.cos(math.pi * (frac - WARMUP) / (1 - WARMUP)))
        z, y = data.batch(BATCH, WINDOW + PAD)
        step += 1
        masks = cold if step % COLD_EVERY == 0 else None
        od.cross_entropy(mix(p, order_rows(z), masks), fold(y[:, PAD:])).backward()
        c1, c2 = LR_MIX * sched / (1 - 0.9**step), 1 / (1 - 0.99**step)
        lrt = LR_TABLE * min(1.0, frac / WARMUP) * (1 - (1 - LR_TAIL) * frac)
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
