"""Are the odometer's prices honest on this machine?

    python tools/calibrate.py                   # prints a table sorted by ns per tick; exit 1 if any row is off
    python tools/calibrate.py --json out.json   # also writes the rows and the machine description

Every row calls the real engine on one shape with one BLAS thread and records nanoseconds per tick, the median of
repeated timed calls. A row is off when its ratio is more than FACTOR from the median of all rows, in either direction.
An entry could buy cheap work from any op priced far below the median, so a pass here is what lets one budget in ticks
mean the same amount of work whatever the op mix.
"""

import argparse
import os
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import verify  # noqa: E402

# numpy reads these once, when it loads
os.environ.update({v: "1" for v in verify.THREAD_VARS})

import numpy as np  # noqa: E402

import odometer as od  # noqa: E402
from odometer import _data, _engine  # noqa: E402

FACTOR = 5
MIN_REPS = 5
ROW_NS = 80_000_000  # wall time to spend on one row once MIN_REPS is reached, setup included

SMALL, MID, LARGE = (4, 16), (64, 256), (1024, 4096)
rng = np.random.default_rng(0)


def floats(shape, positive=False):
    a = rng.standard_normal(shape).astype(np.float32)
    return od.Tensor(np.abs(a) + 0.5 if positive else a)


def leaf(shape, positive=False):
    return od.param(floats(shape, positive))


def ints(shape, high=256):
    return od.Tensor(rng.integers(0, high, size=shape, dtype=np.int32))


def forward(fn, *args):
    return lambda: lambda: fn(*args)


def backward(fn, *args):
    """Time backward() alone. Forward runs in the setup, and a sum reduces the output to the scalar backward() needs."""

    def setup():
        for a in args:
            for t in a if isinstance(a, list) else [a]:
                if isinstance(t, od.Tensor):
                    t.grad = None
        out = fn(*args)
        return (out if out.size == 1 else od.sum(out)).backward

    return setup


def both(rows, name, shape, fn, *args):
    rows.append((name, "fwd", shape, forward(fn, *args)))
    rows.append((name, "bwd", shape, backward(fn, *args)))


def elementwise(rows):
    unary = {
        "neg": (od.neg, False), "abs": (od.abs, False), "relu": (od.relu, False), "exp": (od.exp, False),
        "tanh": (od.tanh, False), "sigmoid": (od.sigmoid, False), "sin": (od.sin, False), "cos": (od.cos, False),
        "log": (od.log, True), "sqrt": (od.sqrt, True), "rsqrt": (od.rsqrt, True),
        "power 2": (lambda x: od.power(x, 2), True), "power 1.5": (lambda x: od.power(x, 1.5), True),
        "softmax": (od.softmax, False), "log_softmax": (od.log_softmax, False),
        "sum": (od.sum, False), "sum axis": (lambda x: od.sum(x, -1), False), "mean": (od.mean, False),
        "max axis": (lambda x: od.max(x, -1), False), "tril": (od.tril, False),
    }
    binary = {"add": od.add, "sub": od.sub, "mul": od.mul, "maximum": od.maximum, "minimum": od.minimum}
    for shape in (SMALL, MID, LARGE):
        for name, (fn, positive) in unary.items():
            both(rows, name, shape, fn, leaf(shape, positive))
        for name, fn in binary.items():
            both(rows, name, shape, fn, leaf(shape), leaf(shape))
        both(rows, "div", shape, od.div, leaf(shape), leaf(shape, True))
        both(rows, "add broadcast", shape, od.add, leaf(shape), leaf(shape[-1:]))
        both(rows, "where", shape, od.where, floats(shape) > 0, leaf(shape), leaf(shape))
        rows.append(("sign", "fwd", shape, forward(od.sign, floats(shape))))
        rows.append(("compare", "fwd", shape, forward(lambda a, b: a > b, floats(shape), floats(shape))))
        rows.append(("to_float", "fwd", shape, forward(od.to_float, ints(shape))))
        a, b = ints(shape), ints(shape) + 1
        for name, fn in {"int add": lambda p, q: p + q, "int mul": lambda p, q: p * q, "int mod": lambda p, q: p % q,
                         "int floordiv": lambda p, q: p // q}.items():
            rows.append((name, "fwd", shape, forward(fn, a, b)))
        rows.append(("zeros", "fwd", shape, forward(od.zeros, shape)))
        rows.append(("ones", "fwd", shape, forward(od.ones, shape)))
        rows.append(("full", "fwd", shape, forward(od.full, shape, 1.5)))
        rows.append(("randn", "fwd", shape, forward(od.randn, shape)))
        rows.append(("rand", "fwd", shape, forward(od.rand, shape)))
        rows.append(("arange", "fwd", shape, forward(od.arange, int(np.prod(shape)))))
        rows.append(("param", "fwd", shape, forward(od.param, floats(shape))))
        p, v = leaf(shape), floats(shape)
        rows.append(("update", "fwd", shape, forward(p.update, v)))


def first_part(x, parts):
    return od.split(x, parts, 0)[0]


def shapes(rows):
    for shape in (SMALL, MID, LARGE):
        x = leaf(shape)
        both(rows, "reshape view", shape, lambda t: t.reshape(-1), x)
        both(rows, "reshape copy", shape, lambda t: t.T.reshape(-1), x)
        both(rows, "reshape copy in back", shape, lambda t, s=shape: t.reshape(s[1], s[0]).T, x)
        both(rows, "transpose", shape, lambda t: t.T, x)
        both(rows, "getitem slice", shape, lambda t: t[1:, :2], x)
    both(rows, "concat 2", LARGE, lambda a, b: od.concat([a, b], 0), leaf(LARGE), leaf(LARGE))
    both(rows, "concat 2", MID, lambda a, b: od.concat([a, b], 0), leaf(MID), leaf(MID))
    both(rows, "concat 2", SMALL, lambda a, b: od.concat([a, b], 0), leaf(SMALL), leaf(SMALL))
    for parts, size in ((16, 1), (1024, 1), (1024, 64), (65536, 1)):
        ts = [leaf((size,)) for _ in range(parts)]
        rows.append((f"concat {parts}", "fwd", (size,), forward(od.concat, ts, 0)))
        rows.append((f"concat {parts}", "bwd", (size,), backward(od.concat, ts, 0)))
    for parts, shape in ((4, (64, 256)), (1024, (1024, 4096)), (65536, (65536,))):
        x = leaf(shape)
        both(rows, f"split {parts}", shape, first_part, x, parts)


def matmuls(rows):
    grid = [
        ((8, 16), (16, 8)), ((64, 64), (64, 64)), ((64, 256), (256, 64)), ((256, 256), (256, 256)),
        ((1024, 1024), (1024, 1024)), ((2048, 2048), (2048, 2048)),  # square, up to where BLAS blocks hit full speed
        ((16, 64, 96), (96, 288)), ((16, 64, 288), (288, 96)), ((16, 4, 64, 24), (16, 4, 24, 64)),  # shapes in a GPT
        ((16, 4, 64, 64), (16, 4, 64, 24)), ((64, 64, 64), (64, 64, 64)), ((256, 16, 16), (256, 16, 16)),
        ((4, 256, 256), (4, 256, 256)),  # batched: one BLAS call per stack element
        ((1 << 20, 1), (1, 1)), ((1024, 1), (1, 1024)), ((1 << 16, 4), (4, 4)),  # thin inner dimension
        ((1, 1 << 16), (1 << 16, 1)), ((1, 1024), (1024, 1024)), ((1024, 1024), (1024, 1)),  # thin outer dimension
    ]
    for sa, sb in grid:
        both(rows, "matmul", f"{sa}@{sb}", od.matmul, leaf(sa), leaf(sb))
    x, w = floats((16, 64, 96)), leaf((96, 288))
    rows.append(("matmul one grad", "bwd", "(16, 64, 96)@(96, 288)", backward(od.matmul, x, w)))
    x, w = leaf((1024, 1024)), floats((1024, 1024))
    rows.append(("matmul one grad", "bwd", "(1024, 1024)@(1024, 1024)", backward(od.matmul, x, w)))


def lookups(rows):
    grid = [
        ((256, 16), (4, 16)), ((256, 64), (16, 64)), ((256, 256), (16, 64)), ((256, 256), (256, 256)),
        ((1 << 12, 64), (64, 256)), ((1 << 16, 1), (1024, 256)), ((1 << 16, 256), (64, 64)),
        ((1 << 20, 8), (1024, 64)), ((1 << 22, 1), (1024, 1024)), ((1 << 20, 64), (1024, 64)),
    ]
    for ts, ixs in grid:
        both(rows, "embedding", f"{ts}[{ixs}]", od.embedding, leaf(ts), ints(ixs, ts[0]))
    for shape in ((4, 8, 16), (16, 64, 256), (256, 256, 256)):
        both(rows, "cross_entropy", shape, od.cross_entropy, leaf(shape), ints(shape[:-1], shape[-1]))
    data = _data.TrainData()
    for b, t in ((1, 8), (16, 64), (256, 1024), (4096, 64)):
        rows.append(("data.batch", "fwd", (b, t), forward(data.batch, b, t)))
    rows.append(("data.batch align", "fwd", (16, 64), forward(lambda: data.batch(16, 64, align=True))))


def walk(rows):
    def chain(x):
        for _ in range(64):
            x = x.reshape(16).reshape(4, 4)
        return x

    rows.append(("backward walk 128 nodes", "bwd", (4, 4), backward(chain, leaf((4, 4)))))


def measure(setup):
    times, ticks = [], None
    begun = time.perf_counter_ns()
    while len(times) < MIN_REPS or time.perf_counter_ns() - begun < ROW_NS:
        work = setup()
        start = _engine._M.used
        t0 = time.perf_counter_ns()
        work()
        times.append(time.perf_counter_ns() - t0)
        spent = _engine._M.used - start
        if ticks is not None and spent != ticks:
            raise SystemExit(f"price changed between identical calls: {ticks} then {spent}")
        ticks = spent
    return statistics.median(times), ticks, len(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="write the rows and the machine description here")
    args = ap.parse_args()

    t0 = time.perf_counter()
    _engine._M.reset(None, 0)
    cases = []
    for group in (elementwise, shapes, matmuls, lookups, walk):
        group(cases)
    results = []
    for op, way, shape, setup in cases:
        ns, ticks, reps = measure(setup)
        results.append({"op": op, "way": way, "shape": str(shape), "ticks": ticks, "ns": ns, "reps": reps,
                        "ns_per_tick": ns / ticks})

    results.sort(key=lambda r: r["ns_per_tick"])
    median = statistics.median(r["ns_per_tick"] for r in results)
    off = []
    print(f"{'op':<24}{'way':<5}{'shape':<30}{'ticks':>13}{'ns':>14}{'ns/tick':>10}{'/median':>9}")
    for r in results:
        r["ratio"] = r["ns_per_tick"] / median
        is_off = not 1 / FACTOR <= r["ratio"] <= FACTOR
        off += [r] if is_off else []
        print(f"{r['op']:<24}{r['way']:<5}{r['shape']:<30}{r['ticks']:>13,}{r['ns']:>14,.0f}{r['ns_per_tick']:>10.3f}"
              f"{r['ratio']:>9.2f}{'  OFF' if is_off else ''}")
    seconds = time.perf_counter() - t0
    print(f"\n{len(results)} rows in {seconds:.0f} s. Median {median:.3f} ns per tick. "
          f"{len(off)} rows are more than a factor of {FACTOR} from it.")
    if args.json:
        doc = {"machine": verify.machine(), "median_ns_per_tick": median, "factor": FACTOR, "seconds": seconds,
               "off": len(off), "rows": results}
        pathlib.Path(args.json).write_text(verify.json.dumps(doc, indent=1))
    sys.exit(1 if off else 0)


if __name__ == "__main__":
    main()
