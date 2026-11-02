"""Hashed context mixing by gradient descent, then a learned in-window PPM stage aware of words and story boundaries.

Extends the previous record, whose table model, gate and blend are kept. Three changes:

1. Low-order tables pay embedding() backward only on some steps. Orders 1, 2 and 3 are live on every 4th step and order 4
   on every 2nd, after the first 15% of the run. Their rows see far more data than they need, and that backward is most
   of the price of a step, so the higher orders get more bytes for the same ticks.
2. Two in-window stages keyed on words rather than on a fixed number of bytes: the case-folded prefix of the current word,
   and that prefix with the previous word (Mahoney's PAQ word models). Words come from a causal scan of the last 16 bytes.
3. A vote from before the last story boundary (a zero byte) in the window counts 0.1 of a vote, since it comes from
   another story with other names.

Measured and dropped: word contexts as tables rather than in-window stages, a mixer-only phase after freezing the tables,
recency-decayed votes, gate features for distance into the story.
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
WORD_K = 16
LOCAL = ((1, 100.0, 0.02), (2, 22.7, 0.022), (3, 7.0, 0.343), (101, 6.0, 0.3), (4, 5.5, 0.226), (102, 3.0, 0.4), (6, 0.85, 0.684), (16, 0.39, 0.6))  # (order, concentration, discount) before the gates
MIXW = (0.51, 0.55)  # logit weight of the softmax and of each stage before the gates  # (order, concentration, discount)
WARMUP = 0.02
HASH_A = 16777619
HASH_B = 2654435761 % (1 << 31)
CLASSES = 76
FLOOR = 1e-7
COLD_EVERY = 16
SAMPLE = (4, 4, 4, 2, 1, 1, 1, 1, 1)
FULL = 0.15
GATE = 16
GATE_BATCH = 4
GATE_T = 256
GATE_LR = 0.02
GATE_SHARE = 0.08
EVAL_T = 512


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


def mix(p, rows, masks=None, live=None):
    parts = [od.embedding(p[f"t{k}"] if live is None or live[k] else p[f"t{k}"].detach(), r) for k, r in enumerate(rows)]
    if masks:
        parts = [e * m for e, m in zip(parts, masks)]
    a = od.relu(od.concat(parts, -1) @ p["up"] + p["bias"])
    return (a * a) @ p["head"]


def word_hashes(z):
    """Causal word contexts. Hashes of: current word prefix; plus previous word; plus the one before."""
    T = z.shape[1]
    ge65 = (z // 65 + 3) // 4
    low = z + (ge65 - (z // 97 + 1) // 2) * 32
    h = z[:, PAD:] * 0
    p1, p2 = h, h
    for n in range(WORD_K, 0, -1):
        a, b = PAD - n + 1, T - n + 1
        L = ge65[:, a:b]
        e = ge65[:, a - 1 : b - 1] * (1 - L)
        k = 1 - e
        p2 = p2 * k + p1 * e
        p1 = p1 * k + h * e
        h = (h * HASH_A + low[:, a:b]) * L
    w0 = h * HASH_B + z[:, PAD:]
    w1 = w0 * HASH_A + p1
    return w0, w1, w1 * HASH_A + p2

def raw_hashes(z):
    """Unreduced hash per in-window order, since a table's modulus would add false matches. (B, T)."""
    T = z.shape[1]
    h = z[:, PAD:] * 0
    out = {}
    for n in range(1, max(i for i, b, d in LOCAL if i < 100) + 1):
        h = h * HASH_A + z[:, PAD - n + 1 : T - n + 1]
        out[n] = h
    w = word_hashes(z)
    return out | {101: w[0], 102: w[1], 103: w[2]}


def identity(n):
    return od.tril(od.ones((n, n))) - od.tril(od.ones((n, n)), -1)


def local_stats(z, x):
    """Per in-window order: votes per class, their total, and how many classes have a vote."""
    B, T = x.shape
    raw = raw_hashes(z)
    mask = od.tril(od.ones((1, T, T)), -1)
    ends = od.sum(od.embedding(identity(256)[:, :1], x), -1, keepdims=True)
    run = od.tril(od.ones((1, T, T))) @ ends
    cross = (run - run.reshape(B, 1, T)) > 0.5
    mask = mask * (1.0 - cross * 0.9)
    follows = od.embedding(identity(CLASSES), fold(x[:, 1:]))
    out = []
    for n, beta, disc in LOCAL:
        d = raw[n].reshape(B, T, 1) - raw[n].reshape(B, 1, T)
        same = (d >= 0) * (d <= 0) * mask
        c = same[:, :, : T - 1] @ follows
        out.append((c, od.sum(c, -1, keepdims=True), od.sum(c > 0.0, -1, keepdims=True)))
    return out


def base_probs(p, z, x):
    return od.softmax(mix(p, order_rows(z), history_masks(x.shape[1])), -1)


def features(p0, stats):
    ent = od.sum(p0 * od.log(p0 + 1e-9), -1, keepdims=True) * -0.25
    top = od.max(p0, -1, keepdims=True)
    parts = [ent, top, od.ones((1, 1, 1)) + p0[:, :, :1] * 0.0]
    for c, t, k in stats:
        seen = t > 0.0
        parts += [od.log(t + 1.0) * 0.25, od.log(k + 1.0) * 0.5, seen, seen * (k <= 1.0), od.max(c, -1, keepdims=True) / (t + 1.0)]
    return od.concat(parts, -1)


def gate_params():
    n = len(LOCAL)
    nf = 3 + 5 * n
    nout = 2 * n + 2
    q = {"g1": od.randn((nf, GATE), 1.0 / math.sqrt(nf)), "g2": od.zeros((GATE, nout)), "g3": od.zeros((nf, nout)), "g4": od.zeros((256, nout))}
    return {k: od.param(v) for k, v in q.items()}


def gates(q, f, x):
    return f @ q["g3"] + od.relu(f @ q["g1"]) @ q["g2"] + od.embedding(q["g4"], x)


def blend(q, p0, stats, x):
    """Back off through the in-window orders with learned concentrations and discounts, then mix all stages as logits."""
    n = len(LOCAL)
    f = features(p0, stats)
    out = gates(q, f, x)
    p = p0
    for i, (order, beta, disc) in enumerate(LOCAL):
        c, t, k = stats[i]
        b = od.exp(out[:, :, i : i + 1]) * beta
        d = od.sigmoid(out[:, :, n + i : n + i + 1] + math.log(disc / (1 - disc)))
        p = (od.relu(c - d) + (k * d + b) * p) / (t + b)
    a0 = out[:, :, 2 * n : 2 * n + 1] + MIXW[0]
    a1 = out[:, :, 2 * n + 1 : 2 * n + 2] + MIXW[1]
    return od.log(p0 + 1e-9) * a0 + od.log(p + 1e-9) * a1


def forward(params, x):
    z = od.concat([x[:, :PAD] * 0, x], 1)
    p0 = base_probs(params, z, x)
    out = od.softmax(blend(params, p0, local_stats(z, x), x), -1)
    eye = identity(256)
    place = od.zeros((CLASSES, 256))
    for b in candidates():
        c = class_of(b)
        place = od.concat([place[:c], place[c : c + 1] + eye[b : b + 1], place[c + 1 :]], 0)
    return od.log(out @ place + FLOOR)


def fit_gate(p, data):
    q = gate_params()
    m = {k: od.zeros(v.shape) for k, v in q.items()}
    s = {k: od.full(v.shape, 1e-8) for k, v in q.items()}
    start, span = od.used(), od.remaining()
    worst, step = 0, 0
    while od.remaining() > 2 * worst + 1_000_000:
        before = od.used()
        frac = (before - start) / span
        z, y = data.batch(GATE_BATCH, GATE_T + PAD)
        x = z[:, PAD:]
        with od.no_grad():
            p0 = base_probs(p, z, x)
            stats = local_stats(z, x)
        for again in range(3):
            od.cross_entropy(blend(q, p0, stats, x), fold(y[:, PAD:])).backward()
            step += 1
            c1, c2 = GATE_LR * 0.5 * (1 + math.cos(math.pi * frac)) / (1 - 0.9**step), 1 / (1 - 0.99**step)
            for k, v in q.items():
                g = v.grad
                m[k] = m[k] * 0.9 + g * 0.1
                s[k] = s[k] * 0.99 + g * g * 0.01
                v.update(v - m[k] * od.rsqrt(s[k] * c2 + 1e-16) * c1)
                v.grad = None
        worst = max(worst, od.used() - before)
    return q


def fit_tables(data, stop):
    p = init_params()
    tables = {f"t{k}": e for k, e in enumerate(EVERY)}
    m = {k: od.zeros(v.shape) for k, v in p.items() if k not in tables}
    s = {k: od.full(v.shape, 1e-8) for k, v in p.items()}
    cold = history_masks(WINDOW)
    start, span = od.used(), od.remaining() - stop
    worst, step = 0, 0
    while od.remaining() - stop > 2 * worst + 1_000_000:
        before = od.used()
        frac = (before - start) / span
        sched = frac / WARMUP if frac < WARMUP else 0.5 * (1 + math.cos(math.pi * (frac - WARMUP) / (1 - WARMUP)))
        z, y = data.batch(BATCH, WINDOW + PAD)
        step += 1
        masks = cold if step % COLD_EVERY == 0 else None
        live = [frac < FULL or step % e == 0 for e in SAMPLE]
        od.cross_entropy(mix(p, order_rows(z), masks, live), fold(y[:, PAD:])).backward()
        c1, c2 = LR_MIX * sched / (1 - 0.9**step), 1 / (1 - 0.99**step)
        lrt = LR_TABLE * min(1.0, frac / WARMUP) * (1 - (1 - LR_TAIL) * frac)
        for k, v in p.items():
            g = v.grad
            if k in tables:
                if g is None or step % tables[k]:
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


def train(data):
    p = fit_tables(data, od.remaining() * GATE_SHARE)
    return p | fit_gate(p, data)
