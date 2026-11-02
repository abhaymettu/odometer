"""A small GPT, the way nanoGPT trains one (Karpathy, after Radford et al. 2019).

Learned positions, LayerNorm, an MLP with a smooth activation, AdamW with weight decay on matrices only, linear warmup
and a cosine decay, gradient clipping. Reads the odometer to fit its schedule to the budget.
"""

import math

import odometer as od

DIM = 96
HEADS = 4
LAYERS = 3
BLOCK = 64  # bytes of context in a training window
BATCH = 16
LR = 3e-3
WARMUP = 0.03  # fraction of the steps
WEIGHT_DECAY = 0.01
CLIP = 1.0
STRIDE = 32  # eval: a new block every STRIDE bytes; each predicts its last STRIDE positions


def init_params():
    std = 0.02
    p = {"tok": od.randn((256, DIM), std), "pos": od.randn((BLOCK, DIM), std), "head": od.randn((DIM, 256), std)}
    for i in range(LAYERS):
        res = std / math.sqrt(2 * LAYERS)
        p[f"qkv{i}"] = od.randn((DIM, 3 * DIM), std)
        p[f"out{i}"] = od.randn((DIM, DIM), res)
        p[f"up{i}"] = od.randn((DIM, 4 * DIM), std)
        p[f"down{i}"] = od.randn((4 * DIM, DIM), res)
        for n in ("g1", "g2"):
            p[f"{n}{i}"] = od.ones((1, 1, DIM))
    p["g_end"] = od.ones((1, 1, DIM))
    return {k: od.param(v) for k, v in p.items()}


def norm(x, gain):
    mu = od.mean(x, -1, keepdims=True)
    v = od.mean((x - mu) ** 2, -1, keepdims=True)
    return (x - mu) * od.rsqrt(v + 1e-5) * gain


def gelu(x):
    return x * od.sigmoid(x * 1.702)


def attend(x, p, i, mask):
    B, T, _ = x.shape
    q, k, v = od.split(x @ p[f"qkv{i}"], 3, -1)
    h = DIM // HEADS
    q, k, v = [t.reshape(B, T, HEADS, h).transpose(0, 2, 1, 3) for t in (q, k, v)]
    w = (q @ k.T) * (1.0 / math.sqrt(h))
    w = od.softmax(od.where(mask, w, -1e9), -1)
    y = (w @ v).transpose(0, 2, 1, 3).reshape(B, T, DIM)
    return y @ p[f"out{i}"]


def block_logits(p, x, mask):
    T = x.shape[1]
    h = od.embedding(p["tok"], x) + p["pos"][:T]
    for i in range(LAYERS):
        h = h + attend(norm(h, p[f"g1{i}"]), p, i, mask)
        h = h + gelu(norm(h, p[f"g2{i}"]) @ p[f"up{i}"]) @ p[f"down{i}"]
    return norm(h, p["g_end"]) @ p["head"]


def causal_mask(T):
    return od.tril(od.ones((T, T)))


def forward(params, x):
    B, C = x.shape
    mask = causal_mask(BLOCK)
    out = [block_logits(params, x[:, : BLOCK], mask)]
    for start in range(STRIDE, C - BLOCK + 1, STRIDE):
        out.append(block_logits(params, x[:, start : start + BLOCK], mask)[:, BLOCK - STRIDE :])
    return od.concat(out, 1)


def train(data):
    p = init_params()
    decay = {k for k, v in p.items() if v.ndim == 2}
    m = {k: od.zeros(v.shape) for k, v in p.items()}
    s = {k: od.zeros(v.shape) for k, v in p.items()}
    mask = causal_mask(BLOCK)

    start = od.used()
    step, step_cost, total_steps = 0, 0, None
    while True:
        before = od.used()
        if total_steps is not None and step >= total_steps:
            return p
        x, y = data.batch(BATCH, BLOCK)
        loss = od.cross_entropy(block_logits(p, x, mask), y)
        loss.backward()

        frac = step / total_steps if total_steps else 0.0
        lr = LR * min(1.0, frac / WARMUP) if frac < WARMUP else LR * 0.5 * (1 + math.cos(math.pi * (frac - WARMUP) / (1 - WARMUP)))
        total = sum(od.sum(v.grad * v.grad) for v in p.values())
        scale = od.minimum(od.rsqrt(total + 1e-6) * CLIP, od.ones(()))
        t = step + 1
        for k, v in p.items():
            g = v.grad * scale
            m[k] = m[k] * 0.9 + g * 0.1
            s[k] = s[k] * 0.999 + g * g * 0.001
            hat = m[k] * (1 / (1 - 0.9**t))
            shat = s[k] * (1 / (1 - 0.999**t))
            delta = hat * od.rsqrt(shat + 1e-16) * lr
            v.update((v * (1 - lr * WEIGHT_DECAY) if k in decay else v) - delta)
            v.grad = None
        step += 1
        step_cost = max(step_cost, od.used() - before)
        if step == 8:
            total_steps = step + int((od.remaining() - step_cost) / (1.02 * step_cost))
