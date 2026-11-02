"""A small GPT, the way nanoGPT trains one (Karpathy's nanoGPT, after Radford et al. 2019, "Language Models are
Unsupervised Multitask Learners").

Learned positions, LayerNorm, GELU, AdamW with weight decay on matrices only, linear warmup and a cosine decay, gradient
clipping. The head is untied from the token table, unlike nanoGPT: at this size the tied table loses about 0.04 bpb.
Everything else follows the price list. Nothing here looks at the wall clock: the schedule is a function of
ticks used, and the loop ends when another step might not fit.

Same maths, cheaper spelling: squares are x * x, not x ** 2 (power costs 2 ticks an element, mul a quarter); the causal
mask is added, not applied with where() (an add has no backward price against a constant); q, k and v are three matmuls
rather than one split, because the backward of a split rebuilds the whole tensor per part.
"""

import math

import odometer as od

DIM = 96
HEADS = 4
LAYERS = 2
BLOCK = 48  # bytes of context in a training window
BATCH = 16
MLP = 4  # hidden width of the feed-forward, as a multiple of DIM
LR = 0.006
BETA2 = 0.99
WARMUP = 0.03  # fraction of the ticks
WEIGHT_DECAY = 0.01
CLIP = 1.0
STRIDE = 16  # eval: a new block every STRIDE bytes; each predicts its last STRIDE positions


def init_params():
    std = 0.02
    p = {"tok": od.randn((256, DIM), std), "pos": od.randn((BLOCK, DIM), std), "head": od.randn((DIM, 256), std)}
    for i in range(LAYERS):
        res = std / math.sqrt(2 * LAYERS)
        for n in ("q", "k", "v"):
            p[f"{n}{i}"] = od.randn((DIM, DIM), std)
        p[f"out{i}"] = od.randn((DIM, DIM), res)
        p[f"up{i}"] = od.randn((DIM, MLP * DIM), std)
        p[f"down{i}"] = od.randn((MLP * DIM, DIM), res)
        p[f"g1{i}"] = od.ones((1, 1, DIM))
        p[f"g2{i}"] = od.ones((1, 1, DIM))
    p["g_end"] = od.ones((1, 1, DIM))
    return {k: od.param(v) for k, v in p.items()}


def norm(x, gain):
    d = x - od.mean(x, -1, keepdims=True)
    return d * od.rsqrt(od.mean(d * d, -1, keepdims=True) + 1e-5) * gain


def gelu(x):
    return x * od.sigmoid(x * 1.702)


def attend(x, p, i, mask):
    B, T, _ = x.shape
    h = DIM // HEADS
    q, k, v = [(x @ p[f"{n}{i}"]).reshape(B, T, HEADS, h).transpose(0, 2, 1, 3) for n in ("q", "k", "v")]
    w = od.softmax((q @ k.T) * (1.0 / math.sqrt(h)) + mask, -1)
    return (w @ v).transpose(0, 2, 1, 3).reshape(B, T, DIM) @ p[f"out{i}"]


def block_logits(p, x, mask):
    T = x.shape[1]
    h = od.embedding(p["tok"], x) + p["pos"][:T]
    for i in range(LAYERS):
        h = h + attend(norm(h, p[f"g1{i}"]), p, i, mask)
        h = h + gelu(norm(h, p[f"g2{i}"]) @ p[f"up{i}"]) @ p[f"down{i}"]
    return norm(h, p["g_end"]) @ p["head"]


def causal_mask(T):
    return (1.0 - od.tril(od.ones((T, T)))) * -1e9


def forward(params, x):
    C = x.shape[1]
    ctx = causal_mask(BLOCK)
    out = [block_logits(params, x[:, :BLOCK], ctx)]
    end = BLOCK
    while end < C:
        n = min(STRIDE, C - end)
        out.append(block_logits(params, x[:, end + n - BLOCK : end + n], ctx)[:, -n:])
        end += n
    return od.concat(out, 1)


def train(data):
    p = init_params()
    decay = {k for k, v in p.items() if v.ndim == 2 and k != "pos"}
    m = {k: od.zeros(v.shape) for k, v in p.items()}
    s = {k: od.zeros(v.shape) for k, v in p.items()}
    mask = causal_mask(BLOCK)
    start, span = od.used(), od.remaining()
    worst, step = 0, 0
    while od.remaining() > 2 * worst + 1_000_000:
        before = od.used()
        frac = (before - start) / span
        lr = LR * (frac / WARMUP if frac < WARMUP else 0.5 * (1 + math.cos(math.pi * (frac - WARMUP) / (1 - WARMUP))))
        x, y = data.batch(BATCH, BLOCK)
        loss = od.cross_entropy(block_logits(p, x, mask), y)
        loss.backward()
        total = sum(od.sum(v.grad * v.grad) for v in p.values())
        scale = od.minimum(od.rsqrt(total + 1e-6) * CLIP, od.ones(()))
        step += 1
        c1, c2 = lr / (1 - 0.9**step), 1 / (1 - BETA2**step)
        for k, v in p.items():
            g = v.grad * scale
            m[k] = m[k] * 0.9 + g * 0.1
            s[k] = s[k] * BETA2 + g * g * (1 - BETA2)
            delta = m[k] * od.rsqrt(s[k] * c2 + 1e-16) * c1
            v.update((v * (1 - lr * WEIGHT_DECAY) if k in decay else v) - delta)
            v.grad = None
        worst = max(worst, od.used() - before)
    return p
