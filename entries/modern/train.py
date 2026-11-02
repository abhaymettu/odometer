"""The modded-nanogpt speedrun recipe (Keller Jordan et al., github.com/KellerJordan/modded-nanogpt; Muon: Jordan et al.
2024, "Muon: An optimizer for hidden layers in neural networks"), ported to the odometer and pruned by the price list.

Muon with Newton-Schulz orthogonalization on hidden matrices, Adam on the embedding and head. RMSNorm without gains
(Zhang & Sennrich 2019), rotary positions (Su et al. 2021), QK-norm (Henry et al. 2020), squared ReLU (So et al. 2021),
untied head, zero-initialized residual outputs and head, trapezoid schedule. All of those earned their ticks at this
budget; three Newton-Schulz steps are the knee, and a window of 48 did at least as well as a deeper or wider model.
"""

import math

import odometer as od

DIM = 96
HEADS = 4
LAYERS = 2
BLOCK = 48
BATCH = 32
MLP = 3
LR_MUON = 0.02
LR_ADAM = 3e-3
MOMENTUM = 0.95
NS_STEPS = 3
RAMP_UP = 0.1  # trapezoid: fractions of the ticks spent going up and coming down
RAMP_DOWN = 0.3
STRIDE = 16
EVAL_BLOCK = 48


def init_params():
    std = 1.0 / math.sqrt(DIM)
    p = {"tok": od.randn((256, DIM), std), "head": od.zeros((DIM, 256))}
    for i in range(LAYERS):
        for n in ("q", "k", "v"):
            p[f"{n}{i}"] = od.randn((DIM, DIM), std)
        p[f"out{i}"] = od.zeros((DIM, DIM))
        p[f"up{i}"] = od.randn((DIM, MLP * DIM), std)
        p[f"down{i}"] = od.zeros((MLP * DIM, DIM))
    return {k: od.param(v) for k, v in p.items()}


def rms_norm(x):
    return x * od.rsqrt(od.mean(x * x, -1, keepdims=True) + 1e-6)


def rotary_tables(T):
    """cos and sin of shape (T, h / 2), as in Su et al. 2021."""
    h = DIM // HEADS
    ts = od.arange(T).reshape(T, 1)
    freqs = od.concat([od.full((1, 1), 10000.0 ** (-2.0 * i / h)) for i in range(h // 2)], 1)
    return od.cos(ts @ freqs), od.sin(ts @ freqs)


def rotate(x, tables):
    c, s = tables
    a, b = od.split(x, 2, -1)
    return od.concat([a * c - b * s, a * s + b * c], -1)


def attend(x, p, i, ctx):
    mask, tables = ctx
    B, T, _ = x.shape
    h = DIM // HEADS
    q, k, v = [(x @ p[f"{n}{i}"]).reshape(B, T, HEADS, h).transpose(0, 2, 1, 3) for n in ("q", "k", "v")]
    q, k = rms_norm(rotate(q, tables)), rms_norm(rotate(k, tables))
    w = od.softmax((q @ k.T) * (1.0 / math.sqrt(h)) + mask, -1)
    return (w @ v).transpose(0, 2, 1, 3).reshape(B, T, DIM) @ p[f"out{i}"]


def block_logits(p, x, ctx):
    h = od.embedding(p["tok"], x)
    for i in range(LAYERS):
        h = h + attend(rms_norm(h), p, i, ctx)
        a = od.relu(rms_norm(h) @ p[f"up{i}"])
        h = h + (a * a) @ p[f"down{i}"]
    return rms_norm(h) @ p["head"]


def context(T):
    return (1.0 - od.tril(od.ones((T, T)))) * -1e9, rotary_tables(T)


def forward(params, x):
    C = x.shape[1]
    ctx = context(EVAL_BLOCK)
    out = [block_logits(params, x[:, :EVAL_BLOCK], ctx)]
    end = EVAL_BLOCK
    while end < C:
        n = min(STRIDE, C - end)
        out.append(block_logits(params, x[:, end + n - EVAL_BLOCK : end + n], ctx)[:, -n:])
        end += n
    return od.concat(out, 1)


def orthogonalize(g):
    """Push all singular values of g towards 1, with the degree-5 polynomial from Jordan's Muon code."""
    x = g * od.rsqrt(od.sum(g * g) + 1e-12)
    for _ in range(NS_STEPS):
        a = x @ x.T
        x = x * 3.4445 + (a * -4.7750 + (a @ a) * 2.0315) @ x
    return x


def trapezoid(frac):
    return min(1.0, frac / RAMP_UP, (1.0 - frac) / RAMP_DOWN)


def train(data):
    p = init_params()
    muon = {k for k in p if k not in ("tok", "head")}
    m = {k: od.zeros(v.shape) for k, v in p.items()}
    s = {k: od.zeros(v.shape) for k, v in p.items() if k not in muon}
    ctx = context(BLOCK)
    start, span = od.used(), od.remaining()
    worst, step = 0, 0
    while od.remaining() > 2 * worst + 1_000_000:
        before = od.used()
        sched = trapezoid((before - start) / span)
        x, y = data.batch(BATCH, BLOCK)
        od.cross_entropy(block_logits(p, x, ctx), y).backward()
        step += 1
        c1, c2 = LR_ADAM * sched / (1 - 0.9**step), 1 / (1 - 0.99**step)
        for k, v in p.items():
            g = v.grad
            if k in muon:
                m[k] = m[k] * MOMENTUM + g
                u = orthogonalize(g + m[k] * MOMENTUM)
                rows, cols = v.shape
                v.update(v - u * (LR_MUON * sched * math.sqrt(max(1.0, rows / cols))))
            else:
                m[k] = m[k] * 0.9 + g * 0.1
                s[k] = s[k] * 0.99 + g * g * 0.01
                v.update(v - m[k] * od.rsqrt(s[k] * c2 + 1e-16) * c1)
            v.grad = None
        worst = max(worst, od.used() - before)
    return p
