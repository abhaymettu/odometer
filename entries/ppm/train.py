"""Prediction by partial matching on hashed counts, with no gradient descent at all.

After Cleary & Witten 1984 (PPM) and Moffat 1990 (PPMC), with the absolute discounting of Kneser & Ney 1995 and the
recursive backoff form of Teh 2006 (hierarchical Pitman-Yor language model). The hashing and the fixed memory follow
Mahoney's PAQ family; pruning rare keys before folding follows the same logic as a counting Bloom filter.

The price list makes this possible. The backward of embedding() is a scatter-add, so the gradient of a sum of table
lookups is a table of counts, at about 30 ticks per byte per order. One 4e10-tick budget counts the whole training file
more than once. Each order keeps a wide table of (context, next byte) keys during training, which is outside the
parameter cap. The wide table is then pruned (counts below TAU go) and folded into the narrow table that is returned.

forward() reads a count per order per candidate byte and backs off from low order to high:
    p_k(y) = (max(c_k(y) - d_k, 0) + (a_k + d_k * t_k) * p_k-1(y)) / (n_k + a_k)
where n_k and t_k are the total and the number of distinct successors. A higher order count can never exceed the count
of the order below it, so taking the minimum removes most false hits from hash collisions. Then the same backoff runs
over counts taken inside the 512-byte window itself, which is the adaptivity online PPM gets from seeing the story so far.
"""

import odometer as od

ORDERS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10)
NARROW = (509, 16381, 131071, 524287, 524287, 524287, 524287, 524287, 524287, 524287)  # kept slots per order, primes
FOLD = (1, 1, 1, 1, 2, 4, 8, 16, 32, 64)  # wide slots per kept slot during training
TAU = (1, 1, 1, 1, 4, 8, 16, 32, 32, 64)  # counts below this do not survive the fold
ALPHA = (1.0, 8.0, 8000.0, 1200.0, 1000.0, 1600.0, 1600.0, 1600.0, 1600.0, 1600.0)
DISC = (0.0, 0.5, 0.6, 0.2, 0.2, 0.7, 0.6, 0.85, 0.85, 0.85)
LOCAL = ((3, 50.0, 0.6), (4, 8.0, 0.6), (6, 3.0, 0.6))  # at eval, (index into ORDERS, concentration, discount) per in-window stage
ROWS = 1024  # windows per counting batch
WINDOW = 2048
MUL_CTX = 1000003
MUL_KEY = 739474771
MUL_NEXT = 1500450271
RARE_BYTES = (128, 147, 148, 152, 153, 156, 157, 195, 226)


def candidates():
    """99.9998% of the training bytes."""
    return (0, 10, *range(32, 123), *RARE_BYTES)


def context_hashes(x, pad):
    """One integer tensor per order: a hash of the n bytes ending at each position of x[:, pad:]."""
    T = x.shape[1]
    h = x[:, pad:] * 0
    out = []
    for n in range(max(ORDERS) + 1):
        if n in ORDERS:
            out.append(h * MUL_KEY)
        h = h * MUL_CTX + x[:, pad - n : T - n] + 1
    return out


def count_step(tables, x, y):
    pad = max(ORDERS)
    next_code = (y[:, pad:] + 1) * MUL_NEXT
    total = None
    for i, h in enumerate(context_hashes(x, pad)):
        slot = (h + next_code) % (NARROW[i] * FOLD[i])
        hits = od.sum(od.embedding(tables[i], slot))
        total = hits if total is None else total + hits
    total.backward()


def train(data):
    tables = [od.param(od.zeros((n * f, 1))) for n, f in zip(NARROW, FOLD)]
    reserve = sum(t.size for t in tables) + 10_000_000
    rows = ROWS
    while rows > 0:
        before = od.used()
        x, y = data.batch(rows, WINDOW)
        count_step(tables, x, y)
        cost_per_row = (od.used() - before) / rows
        rows = min(ROWS, int((od.remaining() - reserve) / (1.05 * cost_per_row)))
    params = {}
    for i, t in enumerate(tables):
        c = t.grad.reshape(FOLD[i], NARROW[i])
        params[f"c{i}"] = od.sum(c * (c >= TAU[i]), 0).reshape(NARROW[i], 1)
    return params


def candidate_codes(x):
    one = x[:1, :1] * 0
    return od.concat([(one + c + 1) * MUL_NEXT for c in candidates()], 1).reshape(1, 1, len(candidates()))


def stage(p, c, a, d):
    total = od.sum(c, -1, keepdims=True)
    kinds = od.sum(c > 0.0, -1, keepdims=True)
    return (od.relu(c - d) + (kinds * d + a) * p) / (total + a)


def window_votes(rows, x):
    """For each position and byte, how often this window already saw that byte follow this context. Causal: s < t."""
    B, T = x.shape
    d = rows.reshape(B, T, 1) - rows.reshape(B, 1, T)
    same = od.tril((d >= 0) * (d <= 0), -1)
    eye = od.tril(od.ones((256, 256))) - od.tril(od.ones((256, 256)), -1)
    return same[:, :, : T - 1] @ od.embedding(eye, x[:, 1:])


def forward(params, x):
    B, T = x.shape
    pad = max(ORDERS)
    codes = candidate_codes(x)
    z = od.concat([x[:, :pad] * 0, x], 1)
    C = len(candidates())
    p = od.full((1, 1, C), 1.0 / C)
    prev = None
    z_rows = []
    for i, h in enumerate(context_hashes(z, pad)):
        slot = (h.reshape(B, T, 1) + codes) % NARROW[i]
        c = od.embedding(params[f"c{i}"], slot).reshape(B, T, C)
        c = c if prev is None else od.minimum(c, prev)
        prev = c
        p = stage(p, c, ALPHA[i], DISC[i])
        z_rows.append(h)
    eye = od.tril(od.ones((256, 256))) - od.tril(od.ones((256, 256)), -1)
    place = od.concat([eye[c : c + 1] for c in candidates()], 0)
    q = p @ place + od.full((1, 1, 256), 1e-9) * (1.0 - od.sum(place, 0))
    for i, beta, disc in LOCAL:
        q = stage(q, window_votes(z_rows[i], x), beta, disc)
    return od.log(q)
