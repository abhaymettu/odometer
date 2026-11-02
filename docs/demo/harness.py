"""The demo track: the current record's code, unchanged, at a small budget on a small excerpt of the data.

The page runs this file in your browser with Pyodide, and `python docs/demo/harness.py` runs it anywhere else. Nothing here is
a score: there are no static rules, no separate processes and no twins, and 32 windows is too few to rank anything. It exists to
show the one property the track is built on. The odometer reading depends on the program alone, so every device stops at the
same tick, however long it took to get there.
"""

import gzip
import json
import pathlib
import sys
import time

import numpy as np

BUDGET = 2_000_000_000
SEED = 11
CONTEXT = 512
PROMPT = "Once upon a time"
SAMPLE_BYTES = 240
TEMPERATURE = 0.7


def excerpt(path):
    raw = pathlib.Path(path).read_bytes()
    return np.frombuffer(gzip.decompress(raw) if str(path).endswith(".gz") else raw, dtype=np.uint8)


def run(entry, train_path, dev_path, on_step=None, on_token=None):
    import odometer as od
    from odometer import _data, _engine

    # In a browser the excerpt arrives gzipped, so it is unpacked once and handed to the engine's own data class.
    unpacked = pathlib.Path(train_path).with_suffix("") if str(train_path).endswith(".gz") else pathlib.Path(train_path)
    if not unpacked.exists():
        unpacked.write_bytes(excerpt(train_path).tobytes())

    class Data(_data.TrainData):
        def batch(self, *a, **k):
            if on_step:
                on_step(_engine._M.used, last_loss[0])
            return super().batch(*a, **k)

    last_loss = [None]
    plain_loss = od.cross_entropy

    def noting_loss(*a, **k):
        out = plain_loss(*a, **k)
        last_loss[0] = float(out._d) / np.log(2)
        return out

    module = {}
    exec(compile(pathlib.Path(entry).read_text(encoding="utf-8"), "train.py", "exec"), module)
    _engine._M.reset(BUDGET, SEED)
    data = Data(unpacked)
    od.cross_entropy = noting_loss
    t0 = time.perf_counter()
    try:
        params = module["train"](data)
    finally:
        od.cross_entropy = plain_loss
    train_seconds = time.perf_counter() - t0
    reading = _engine._M.used
    arrays = {k: v._d for k, v in params.items()}

    _engine._M.reset(None, SEED, mode="eval")
    t0 = time.perf_counter()
    windows = excerpt(dev_path).reshape(-1, CONTEXT + 1)
    nll = 0.0
    for start in range(0, len(windows), 16):
        rows = windows[start : start + 16]
        logits = module["forward"]({k: od.Tensor(a) for k, a in arrays.items()}, od.Tensor(rows[:, :CONTEXT].astype(np.int32)))
        log_p = logits._d - np.logaddexp.reduce(logits._d, axis=-1, keepdims=True)
        nll -= np.take_along_axis(log_p, rows[:, 1:, None].astype(np.int64), axis=-1).sum()
    bpb = float(nll / (len(windows) * CONTEXT) / np.log(2))
    eval_seconds = time.perf_counter() - t0

    rng = np.random.default_rng([SEED, 3])
    text = list(PROMPT.encode())
    for _ in range(SAMPLE_BYTES):
        tail = text[-96:]
        x = np.zeros((1, 96), dtype=np.int32)
        x[0, -len(tail) :] = tail
        _engine._M.reset(None, SEED, mode="eval")
        logits = module["forward"]({k: od.Tensor(a) for k, a in arrays.items()}, od.Tensor(x))._d[0, -1] / TEMPERATURE
        p = np.exp(logits - logits.max())
        text.append(int(rng.choice(256, p=p / p.sum())))
        if on_token:
            on_token(bytes(text).decode("utf-8", "replace"))
    return {"reading": reading, "budget": BUDGET, "bpb": bpb, "train_seconds": train_seconds, "eval_seconds": eval_seconds,
            "ns_per_tick": train_seconds * 1e9 / reading, "text": bytes(text).decode("utf-8", "replace")}


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent.parent))
    report = run(here / "train.py", here / "train.bin.gz", here / "dev.bin")
    print(json.dumps(report, indent=1))
