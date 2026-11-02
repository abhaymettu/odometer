"""Hashed context mixing by gradient descent, then a learned in-window PPM stage, with a gate fitted on fresh data.

Built on the previous record. Its model and its table optimizer are kept. Measured on the dev split, and what changed:

1. The three highest-order tables (8, 10, 12) go from 8 to 4 floats per row with twice the rows. Their contexts have few
   successors, so they need less width, the backward of embedding() costs half, and there are fewer collisions.
2. The gate was short of fresh bytes: on cached tables, 13% and 32% of the budget gave -0.002 and -0.005 bpb against 8%.
   But the tables are short of ticks too, and taking the 13% from them lost more than the gate won. So the gate step got
   cheaper instead. Byte matches in the window come from one equality matrix multiplied along its diagonals rather than
   hash comparisons per order; window constants are built once; a stage multiplies by a reciprocal rather than dividing a
   row; the in-window stages of order 1 and 16 are gone, which cost nothing measurable. That is 348 gate steps, not 263.

Measured and dropped: word-context tables in the softmax, as a swap or an addition (they cost more in ticks and rows than
they gave); an order-16 table; thinner rows at order 6 or at low orders; wider low orders; batches of 256 and 512; 48
hidden units; training the back-off on log p(target) alone, which is far cheaper and much worse.
"""

import math

import odometer as od

ORDERS = (1, 2, 3, 4, 5, 6, 8, 10, 12)
ROWS = (256, 8191, 16381, 32749, 75989, 75989, 203999, 203999, 203999)
EMBED = (8, 8, 8, 8, 8, 8, 4, 4, 4)
EVERY = (1, 1, 1, 4, 8, 8, 16, 16, 16)
HIDDEN = 64
BATCH = 128
WINDOW = 64
LR_TABLE = 0.3
LR_TAIL = 0.4  # the table rate falls linearly to this fraction of LR_TABLE over the run
LR_MIX = 1.2e-2
PAD = 32
WORD_K = 16
LOCAL = ((2, 22.7, 0.022), (3, 7.0, 0.343), (101, 6.0, 0.3), (4, 5.5, 0.226), (102, 3.0, 0.4), (6, 0.85, 0.684))
MIXW = (0.51, 0.55)  # logit weight of the softmax and of each stage before the gates  # (order, concentration, discount)
WARMUP = 0.02
HASH_A = 16777619
HASH_B = 2654435761 % (1 << 31)
CLASSES = 76
FLOOR = 1e-7
COLD_EVERY = 16
SAMPLE = (4, 4, 4, 2, 1, 1, 1, 1, 1)
FULL = 0.15
GATE_BATCH = 4
GATE_T = 256
GATE_LR = 0.02
GATE_SHARE = 0.08
GATE_TAB = 0.1
GATE_TABLES = ("g4", "g5")
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
    p = {f"t{k}": od.zeros((r, EMBED[k])) for k, r in enumerate(ROWS)}
    n = sum(EMBED)
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

def identity(n):
    return od.tril(od.ones((n, n))) - od.tril(od.ones((n, n)), -1)


def window_constants(T):
    """What local_stats needs that depends only on the window length; fit_gate builds it once."""
    below = od.tril(od.ones((1, T, T)))
    return below, od.tril(od.ones((1, T, T)), -1), identity(256)[:, :1], identity(CLASSES), history_masks(T)


def local_stats(z, x, wc):
    """Per in-window order: votes per class, their total, and how many classes have a vote."""
    B, T = x.shape
    K = 15
    below, mask, first, classes, unused = wc
    ends = od.sum(od.embedding(first, x), -1, keepdims=True)
    run = below @ ends
    cross = (run - run.reshape(B, 1, T)) > 0.5
    mask = mask * (1.0 - cross * 0.9)
    y = z[:, PAD - K :]
    n = T + K
    d = y.reshape(B, n, 1) - y.reshape(B, 1, n)
    e = (d >= 0) * (d <= 0)
    m1 = e[:, K:, K:] * mask
    m2 = m1 * e[:, K - 1 : n - 1, K - 1 : n - 1]
    m3 = m2 * e[:, K - 2 : n - 2, K - 2 : n - 2]
    m4 = m3 * e[:, K - 3 : n - 3, K - 3 : n - 3]
    m6 = m4 * e[:, K - 4 : n - 4, K - 4 : n - 4] * e[:, K - 5 : n - 5, K - 5 : n - 5]
    r = e[:, 1:, 1:] * e[:, : n - 1, : n - 1]
    r = r[:, 2:, 2:] * r[:, : n - 3, : n - 3]
    r = r[:, 4:, 4:] * r[:, : n - 7, : n - 7]
    m16 = r[:, 8:, 8:] * r[:, : T, : T] * mask
    w = word_hashes(z)
    follows = od.embedding(classes, fold(x[:, 1:]))
    out = []
    for n, beta, disc in LOCAL:
        if n > 100:
            h = w[n - 101]
            d = h.reshape(B, T, 1) - h.reshape(B, 1, T)
            same = (d >= 0) * (d <= 0) * mask
        else:
            same = {1: m1, 2: m2, 3: m3, 4: m4, 6: m6, 16: m16}[n]
        c = same[:, :, : T - 1] @ follows
        out.append((c, od.sum(c, -1, keepdims=True), od.sum(c > 0.0, -1, keepdims=True)))
    return out


def base_probs(p, z, masks):
    return od.softmax(mix(p, order_rows(z), masks), -1)


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
    q = {"g3": od.zeros((nf, nout)), "g4": od.zeros((256, nout)), "g5": od.zeros((2039, nout))}
    return {k: od.param(v) for k, v in q.items()}


def gates(q, f, x):
    return f @ q["g3"] + od.embedding(q["g4"], x) + od.embedding(q["g5"], (x * 251 + od.concat([x[:, :1] * 0, x[:, :-1]], 1)) % 2039)


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
        p = (od.relu(c - d) + (k * d + b) * p) * (1.0 / (t + b))
    a0 = out[:, :, 2 * n : 2 * n + 1] + MIXW[0]
    a1 = out[:, :, 2 * n + 1 : 2 * n + 2] + MIXW[1]
    return od.log(p0 + 1e-9) * a0 + od.log(p + 1e-9) * a1


def forward(params, x):
    z = od.concat([x[:, :PAD] * 0, x], 1)
    wc = window_constants(x.shape[1])
    p0 = base_probs(params, z, wc[4])
    out = od.softmax(blend(params, p0, local_stats(z, x, wc), x), -1)
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
    wc = window_constants(GATE_T)
    start, span = od.used(), od.remaining()
    worst, step = 0, 0
    while od.remaining() > 2 * worst + 1_000_000:
        before = od.used()
        frac = (before - start) / span
        z, y = data.batch(GATE_BATCH, GATE_T + PAD)
        x = z[:, PAD:]
        with od.no_grad():
            p0 = base_probs(p, z, wc[4])
            stats = local_stats(z, x, wc)
        od.cross_entropy(blend(q, p0, stats, x), fold(y[:, PAD:])).backward()
        step += 1
        c1, c2 = GATE_LR * 0.5 * (1 + math.cos(math.pi * frac)) / (1 - 0.9**step), 1 / (1 - 0.99**step)
        for k, v in q.items():
            g = v.grad
            if k in GATE_TABLES:
                s[k] = s[k] + g * g
                v.update(v - g * od.rsqrt(s[k]) * (GATE_TAB * (1 - 0.7 * frac)))
                v.grad = None
                continue
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
