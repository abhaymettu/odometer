"""Rerun an entry on fresh seeds and score it. This file is the referee.

    python verify.py entries/gpt/train.py                         # 8 fresh seeds, scored on the test split
    python verify.py entries/gpt/train.py --seeds 2               # 2 fresh seeds
    python verify.py entries/mine/train.py --seeds 11,12 --split dev
    python verify.py entries/mine/train.py --against records/current.json
    python verify.py entries/mine/train.py --smoke                # seconds, for checking that an entry runs

What a run guarantees:
  1. The source passes static rules. It imports only odometer and math, and it can name only the engine's public API,
     math, and plain container methods. No try, class, global, match, decorator or attribute assignment (except
     x.grad = None), and no module-level state beyond constants. Literals and syntax nodes are rationed, so a table
     cannot be smuggled in the source.
  2. train() runs in its own process with one BLAS thread, under the training budget, with the verifier's seed. That
     process never loads held-out bytes. A run that lets BudgetExhausted escape is void.
  3. Python work outside the engine is counted, not timed: entry lines and jumps are capped. One C call counts once
     however long it runs (a sum over a long range, big-integer arithmetic); that work has only the rationed item()
     reads to learn from, and the wall limit ends it.
  4. Scoring runs in a second, fresh process. forward() sees held-out inputs, never targets, and its compute per byte
     is capped. Every call gets a fresh params dict and fresh tensors, so no state survives between calls. A quarter
     of the windows get a twin whose bytes after a random cut come from another window; a shuffle puts nearly all twins
     in other batches. Logits before the cut must match, which catches forward() looking ahead or across rows.
  5. Seeds are drawn fresh unless given, and written to the log, so anyone can rerun the exact runs.
"""

import argparse
import ast
import builtins
import hashlib
import json
import math
import os
import pathlib
import platform
import secrets
import subprocess
import sys
import tempfile
import time
import types
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import stats  # noqa: E402

TRACK = {
    "version": "odometer-v3",
    "train_budget": 40_000_000_000,  # ticks for everything train() does; about 80 s on a 4-core VM, 25 s on an M4
    "eval_ticks_per_byte": 50_000,  # forward() price cap per scored byte
    "param_cap": 1 << 22,  # floats in the returned parameters, plus characters in their names
    "source_cap": 64 * 1024,  # bytes of train.py
    "literal_cap": 4096,  # bytes of literals, keyword names and function names in train.py
    "node_cap": 8192,  # syntax nodes in train.py; a table spelled in variable names has no literals to count
    "python_lines": 20_000_000,  # entry lines and jumps per process
    "eval_context": 512,
    "eval_batch": 16,
    "eval_windows": 1024,  # windows of eval_context bytes per held-out split
    "twin_fraction": 0.25,
    "twin_positions": 32,
    "twin_tolerance": 1e-4,  # max |log p| difference between twins, in nats; a leak this small gains 1.4e-4 bits per byte
    "item_cap": 20_000,  # item() reads in train()
    "wall_limit_seconds": 3600,  # only catches hangs; it never decides a score
}
SMOKE = {"train_budget": 2_000_000_000, "eval_windows": 32, "wall_limit_seconds": 300}
RECORD_ALPHA = 0.01
RECORD_MARGIN = 0.002  # bits per byte a new record must clear on top of the significance test
LOW_VARIANCE_RATIO = 0.3  # seed SD below this fraction of the incumbent's gets flagged for a human look
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")

# ---------------------------------------------------------------- static rules

ALLOWED_IMPORTS = {"odometer", "math"}
ALLOWED_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "float", "int", "isinstance", "len", "list", "max",
    "min", "pow", "print", "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "True", "False", "None",
}
# math without its functions over sequences and big integers, which do unbounded numeric work in one counted line
MATH_NAMES = {n for n in dir(math) if not n.startswith("_")} - {
    "comb", "dist", "factorial", "fsum", "gcd", "hypot", "isqrt", "lcm", "perm", "prod", "sumprod",
}
CONTAINER_ATTRS = {
    "append", "extend", "insert", "pop", "remove", "reverse", "sort", "index", "count", "clear", "copy", "items", "keys",
    "values", "get", "setdefault", "update", "join", "split", "strip", "startswith", "endswith",
}
BANNED_NODES = (
    ast.ClassDef, ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await, ast.Try, ast.TryStar, ast.Match,
    ast.Global, ast.Nonlocal, ast.Delete, ast.TypeAlias,
)


def allowed_attrs():
    import odometer as od

    engine = {n for n in dir(od) if not n.startswith("_")} - {"Tensor"}
    tensor = {n for n in dir(od.Tensor) if not n.startswith("_")}
    return engine, engine | tensor | {"batch", "num_bytes"} | MATH_NAMES | CONTAINER_ATTRS


def is_constant_expr(node):
    return all(isinstance(n, (ast.Constant, ast.Tuple, ast.UnaryOp, ast.BinOp, ast.Name, ast.Load, ast.operator, ast.unaryop))
               for n in ast.walk(node))


def literal_bytes(tree):
    """Everything in the source an entry can read back as data. dict(name=1) and str(function) turn names into strings."""
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef)) and n.body and isinstance(n.body[0], ast.Expr)
                  and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str)}

    def size(v):
        if isinstance(v, str):
            return len(v.encode("utf-8", "surrogatepass"))
        return len(v) if isinstance(v, bytes) else max(8, v.bit_length() // 8) if isinstance(v, int) else 8

    total = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and id(n) not in docstrings:
            total += size(n.value)
        elif isinstance(n, ast.keyword) and n.arg:
            total += size(n.arg)
        elif isinstance(n, ast.FunctionDef):
            total += size(n.name)
    return total


def static_problems(src, track=TRACK):
    if len(src.encode()) > track["source_cap"]:
        return [f"source is over {track['source_cap']} bytes"]
    try:
        tree = ast.parse(src)
    except (SyntaxError, MemoryError, RecursionError) as e:
        return [f"syntax error: {e}"]
    engine, attrs = allowed_attrs()
    forbidden = set(dir(builtins)) - ALLOWED_BUILTINS
    nodes = list(ast.walk(tree))
    clears_grad = {id(t) for n in nodes if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and n.value.value is None
                   for t in n.targets if isinstance(t, ast.Attribute) and t.attr == "grad"}
    problems = []
    for node in nodes:
        line = getattr(node, "lineno", 0)
        if isinstance(node, BANNED_NODES):
            problems.append(f"line {line}: {type(node).__name__} is not allowed")
        elif isinstance(node, ast.Import):
            problems += [f"line {line}: import {a.name}" for a in node.names if a.name not in ALLOWED_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_IMPORTS:
                problems.append(f"line {line}: from {'.' * node.level}{node.module or ''} import")
            else:
                names = engine if node.module == "odometer" else MATH_NAMES
                problems += [f"line {line}: from {node.module} import {a.name}" for a in node.names if a.name not in names]
        elif isinstance(node, ast.alias):
            if node.asname and node.asname.startswith("_"):  # import math as __builtins__ swaps the builtins table
                problems.append(f"line {line}: {node.asname} starts with an underscore")
        elif isinstance(node, ast.Attribute):
            if node.attr not in attrs:
                problems.append(f"line {line}: .{node.attr} is not on the allowed list")
            if not isinstance(node.ctx, ast.Load) and id(node) not in clears_grad:
                problems.append(f"line {line}: assigning to .{node.attr}")
        elif isinstance(node, ast.Name):
            if node.id.startswith("_") and node.id != "_":
                problems.append(f"line {line}: {node.id} starts with an underscore")
            elif node.id in forbidden:
                problems.append(f"line {line}: builtin {node.id} is not on the allowed list")
        elif isinstance(node, (ast.FunctionDef, ast.Lambda)):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("_"):
                    problems.append(f"line {line}: {node.name} starts with an underscore")
                if node.decorator_list:  # a decorator is a call at module level, so it could build state that persists
                    problems.append(f"line {line}: decorators are not allowed")
                if node.returns and not is_constant_expr(node.returns):
                    problems.append(f"line {line}: an annotation must be a constant")
            if not all(is_constant_expr(d) for d in [*node.args.defaults, *node.args.kw_defaults] if d is not None):
                problems.append(f"line {line}: a default argument must be a constant")
        elif isinstance(node, ast.arg) and node.annotation and not is_constant_expr(node.annotation):
            problems.append(f"line {line}: an annotation must be a constant")  # annotations run once, at def time
    for node in tree.body:
        fine = isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        fine |= isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        fine |= (isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets) and is_constant_expr(node.value))
        if not fine:
            problems.append(f"line {node.lineno}: module level allows only imports, defs and constants")
    if literal_bytes(tree) > track["literal_cap"]:
        problems.append(f"literals total {literal_bytes(tree)} bytes, over the cap of {track['literal_cap']}")
    if len(nodes) > track["node_cap"]:
        problems.append(f"source has {len(nodes)} syntax nodes, over the cap of {track['node_cap']}")
    top = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    problems += [f"no top-level {f}()" for f in ("train", "forward") if f not in top]
    return problems


def load_entry(path, track):
    """Execute the entry with a builtins table that has no way out of the engine."""
    src = pathlib.Path(path).read_text(encoding="utf-8")
    problems = static_problems(src, track)
    if problems:
        raise SystemExit("static rules failed:\n  " + "\n  ".join(problems))

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in ALLOWED_IMPORTS:
            raise ImportError(f"import of {name} is not allowed")
        return builtins.__import__(name, globals, locals, fromlist, level)

    safe = {k: getattr(builtins, k) for k in ALLOWED_BUILTINS}
    safe["__import__"] = guarded_import
    module = types.ModuleType("entry")
    module.__dict__["__builtins__"] = safe
    exec(compile(src, str(path), "exec"), module.__dict__)
    return module


def peak_rss_mb():
    try:
        import resource
    except ImportError:  # Windows
        return None
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024 ** (2 if sys.platform == "darwin" else 1)


def count_python_work(entry, cap, violation):
    """Deterministic fence on Python outside the engine: every line and jump of the entry costs one count."""
    mon, events = sys.monitoring, sys.monitoring.events
    seen = [0]

    def on_event(code, *rest):
        if code.co_filename != entry:
            return mon.DISABLE
        seen[0] += 1
        if seen[0] > cap:
            raise violation(f"more than {cap} Python lines and jumps in the entry")

    mon.use_tool_id(mon.PROFILER_ID, "odometer")
    mon.register_callback(mon.PROFILER_ID, events.LINE, on_event)
    mon.register_callback(mon.PROFILER_ID, events.JUMP, on_event)
    mon.set_events(mon.PROFILER_ID, events.LINE | events.JUMP)
    return seen


# ---------------------------------------------------------------- the two child processes


def child_train(entry, seed, out):
    import numpy as np

    import odometer as od
    from odometer import _data, _engine

    out = pathlib.Path(out)
    track = json.loads((out / "track.json").read_text())
    _engine._M.reset(track["train_budget"], seed, item_cap=track["item_cap"])
    data = _data.TrainData()
    lines = count_python_work(entry, track["python_lines"], od.RuleViolation)
    t0 = time.perf_counter()
    status, params = "ok", None
    try:
        params = load_entry(entry, track).train(data)
    except od.BudgetExhausted as e:
        status = f"void: BudgetExhausted escaped train() ({e})"
    except od.RuleViolation as e:
        status = f"void: rule violation: {e}"
    seconds = time.perf_counter() - t0
    sys.monitoring.set_events(sys.monitoring.PROFILER_ID, 0)
    if status == "ok":
        if not isinstance(params, dict) or not all(isinstance(k, str) and type(v) is od.Tensor for k, v in params.items()):
            status = "void: train() must return a dict of name -> Tensor"
        elif any(v.is_int for v in params.values()):
            status = "void: parameters must be float tensors"
        elif sum(v.size + len(k) for k, v in params.items()) > track["param_cap"]:
            status = f"void: more than {track['param_cap']} parameters, with one more per character of their names"
        else:
            np.savez(out / "params.npz", **{f"p{i}": v._d for i, v in enumerate(params.values())})
            (out / "names.json").write_text(json.dumps(list(params)))
    report = {"status": status, "train_ticks": _engine._M.used, "train_seconds": seconds, "item_reads": _engine._M.items,
              "train_python_lines": lines[0], "peak_rss_mb": peak_rss_mb(), "params": sum(v.size for v in params.values()) if status == "ok" else None}
    (out / "train.json").write_text(json.dumps(report))


def child_eval(entry, seed, split, out):
    import numpy as np

    import odometer as od
    from odometer import _data, _engine

    out = pathlib.Path(out)
    track = json.loads((out / "track.json").read_text())
    C, B = track["eval_context"], track["eval_batch"]
    windows = _data.heldout_windows(split, C, track["eval_windows"])
    n = len(windows)
    rng = np.random.default_rng([seed, 2])
    twins = np.sort(rng.choice(n, int(n * track["twin_fraction"]), replace=False))
    cuts = rng.integers(1, C, size=twins.size)
    donors = windows[rng.integers(0, n, size=twins.size)]
    rows = np.concatenate([windows[:, :C], windows[twins, :C]])
    probe = []
    for j, (cut, donor) in enumerate(zip(cuts, donors)):
        rows[n + j, cut:] = donor[cut:C]
        last = np.arange(max(0, cut - 8), cut)  # a look-ahead leak shows first just before the cut
        probe.append(np.union1d(last, rng.choice(cut, min(cut, track["twin_positions"]), replace=False)))
    probe_rows = {int(r): j for j, r in enumerate(twins)} | {n + j: j for j in range(twins.size)}
    order = rng.permutation(len(rows))

    scored_bytes = len(rows) * C
    _engine._M.reset(track["eval_ticks_per_byte"] * scored_bytes, seed, mode="eval")
    lines = count_python_work(entry, track["python_lines"], od.RuleViolation)
    t0 = time.perf_counter()
    status = "ok"
    nll = np.zeros(n)
    seen = {}
    try:
        module = load_entry(entry, track)
        names = json.loads((out / "names.json").read_text())
        stored = np.load(out / "params.npz")
        arrays = {name: stored[f"p{i}"] for i, name in enumerate(names)}
        for a in arrays.values():
            a.setflags(write=False)
        for start in range(0, len(rows), B):
            idx = order[start : start + B]
            logits = module.forward({k: od.Tensor(a) for k, a in arrays.items()}, od.Tensor(rows[idx].astype(np.int32)))
            if type(logits) is not od.Tensor or logits.is_int or logits.shape != (len(idx), C, od.VOCAB):
                raise od.RuleViolation(f"forward() must return float logits of shape (batch, {C}, {od.VOCAB})")
            z = logits._d.astype(np.float64)
            if not np.isfinite(z).all():
                raise od.RuleViolation("forward() returned non-finite logits")
            lp = z - z.max(-1, keepdims=True)
            lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
            for b, r in enumerate(idx):
                if r < n:
                    nll[r] = -lp[b, np.arange(C), windows[r, 1:]].sum()
                if int(r) in probe_rows:
                    j = probe_rows[int(r)]
                    seen.setdefault(j, []).append(lp[b, probe[j]])
    except od.BudgetExhausted:
        status = f"void: forward() cost more than {track['eval_ticks_per_byte']:,} ticks per byte"
    except od.RuleViolation as e:
        status = f"void: rule violation: {e}"
    sys.monitoring.set_events(sys.monitoring.PROFILER_ID, 0)
    twin_diff = max((float(np.abs(a - b).max()) for a, b in seen.values()), default=0.0) if status == "ok" else None
    if status == "ok" and twin_diff > track["twin_tolerance"]:
        status = f"void: twin check failed (logits before the cut moved by {twin_diff:.2e} nats)"
    report = {"status": status, "split": split, "bpb": float(nll.sum() / (n * C) / np.log(2)) if status == "ok" else None,
              "eval_ticks_per_byte": _engine._M.used / scored_bytes, "twin_max_diff": twin_diff,
              "eval_python_lines": lines[0], "eval_seconds": time.perf_counter() - t0}
    (out / "eval.json").write_text(json.dumps(report))


# ---------------------------------------------------------------- the parent


def run_seed(entry, seed, split, track, keep):
    env = {"PATH": os.environ.get("PATH", ""), **{v: "1" for v in THREAD_VARS}}
    with tempfile.TemporaryDirectory() as tmp:
        (pathlib.Path(tmp) / "track.json").write_text(json.dumps(track))
        result = {"seed": seed}
        for stage, extra in (("train", []), ("eval", [split])):
            cmd = [sys.executable, "-I", str(ROOT / "verify.py"), f"--child-{stage}", str(entry), str(seed), *extra, tmp]
            log = keep / f"seed-{seed}-{stage}.log"
            try:
                with open(log, "wb") as sink:
                    subprocess.run(cmd, cwd=tmp, env=env, stdout=sink, stderr=subprocess.STDOUT, timeout=track["wall_limit_seconds"])
            except subprocess.TimeoutExpired:
                result["status"] = f"void: {stage} process passed the wall limit"
                return result
            path = pathlib.Path(tmp) / f"{stage}.json"
            if not path.exists():
                result["status"] = f"void: {stage} process crashed (see {log.name})"
                return result
            result |= json.loads(path.read_text())
            if result["status"] != "ok":
                return result
        return result


def machine():
    import numpy as np

    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    return {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "numpy": np.__version__, "blas": f"{blas.get('name')} {blas.get('version')}", "cpus": os.cpu_count()}


def code_hash():
    h = hashlib.sha256()
    for p in [ROOT / "verify.py", *sorted((ROOT / "odometer").glob("*.py"))]:
        h.update(p.read_bytes())
    return h.hexdigest()


def run_batch(entry, seeds, split, track, logs, jobs, label=""):
    with ThreadPoolExecutor(jobs) as pool:
        results = []
        for r in pool.map(lambda s: run_seed(entry, s, split, track, logs), seeds):
            results.append(r)
            if r["status"] == "ok":
                print(f"  seed {r['seed']:>10}  {split} bpb {r['bpb']:.4f}  train {r['train_ticks']:.3e} ticks  "
                      f"{r['train_seconds']:6.1f} s  eval {r['eval_ticks_per_byte']:,.0f} ticks/byte  "
                      f"twins {r['twin_max_diff']:.1e}{label}", flush=True)
            else:
                print(f"  seed {r['seed']:>10}  {r['status']}{label}", flush=True)
    return results


def fresh_seeds(n):
    return [secrets.randbits(31) for _ in range(n)]


def judge(bpbs, record_bpbs):
    """Does this batch beat the record by more than the margin? Welch, one-sided, on the verifier's own seeds."""
    w = stats.welch(bpbs, [b - RECORD_MARGIN for b in record_bpbs])
    plain = stats.welch(bpbs, record_bpbs)
    return {"gain": -plain["diff"], "ci95": plain["ci95"], "df": w["df"], "p_beyond_margin": w["p_one_sided"],
            "passes": w["p_one_sided"] < RECORD_ALPHA}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entry")
    ap.add_argument("--seeds", default="8", help="a count of fresh seeds, or a comma list to rerun exact seeds ('11,' is seed 11)")
    ap.add_argument("--split", default="test", choices=("dev", "test"))
    ap.add_argument("--jobs", type=int, default=0, help="seeds in parallel (default: min(seeds, cpus))")
    ap.add_argument("--against", help="a record log; tests whether this entry beats it, then confirms on new seeds")
    ap.add_argument("--smoke", action="store_true", help="a tiny budget and a few windows; the number is not a score")
    ap.add_argument("--out", help="where to write the log (default runs/<entry>-<time>.json)")
    args = ap.parse_args()

    track = {**TRACK, **SMOKE} if args.smoke else TRACK
    entry = pathlib.Path(args.entry).resolve()
    src = entry.read_text(encoding="utf-8")
    problems = static_problems(src, track)
    if problems:
        print("static rules failed:\n  " + "\n  ".join(problems))
        sys.exit(2)
    seeds = [int(s) for s in args.seeds.split(",") if s] if "," in args.seeds else fresh_seeds(int(args.seeds))
    name = entry.parent.name if entry.name == "train.py" else entry.stem
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = pathlib.Path(args.out) if args.out else ROOT / "runs" / f"{name}-{stamp}.json"
    logs = out.with_suffix("")
    logs.mkdir(parents=True, exist_ok=True)
    jobs = args.jobs or max(1, min(len(seeds), os.cpu_count() or 1))
    print(f"{name}: {len(seeds)} seeds on {args.split}, {jobs} at a time, budget {track['train_budget']:.1e} ticks"
          + (" (smoke run: not a score)" if args.smoke else ""), flush=True)

    t0 = time.time()
    results = run_batch(entry, seeds, args.split, track, logs, jobs)
    ok = [r["bpb"] for r in results if r["status"] == "ok"]
    log = {"entry": str(entry.relative_to(ROOT)) if entry.is_relative_to(ROOT) else entry.name,
           "sha256": hashlib.sha256(src.encode()).hexdigest(), "code_sha256": code_hash(), "track": track,
           "split": args.split, "smoke": args.smoke, "machine": machine(), "started": stamp, "seeds": results}
    if len(ok) == len(results) and len(ok) >= 2:
        m, half = stats.ci95(ok)
        log["summary"] = {"mean_bpb": m, "ci95": half, "sd": stats.mean_sd(ok)[1], "n": len(ok)}
        print(f"  mean {m:.4f} +- {half:.4f} bpb (95% CI, n={len(ok)}, sd {log['summary']['sd']:.4f})")
    elif len(ok) != len(results):
        print("  void: at least one seed failed, so the entry has no score")
    elif ok:
        print(f"  {ok[0]:.4f} bpb on one seed (use at least 2 for an interval)")

    if args.against and "summary" in log:
        inc = json.loads(pathlib.Path(args.against).read_text())
        if inc["split"] != args.split:
            raise SystemExit("the record was scored on a different split")
        incumbent = [r["bpb"] for r in inc.get("confirmation", inc["seeds"])]
        stage1 = judge(ok, incumbent)
        verdict = {"against": args.against, "margin": RECORD_MARGIN, "alpha": RECORD_ALPHA, "stage1": stage1, "record": False}
        print(f"  gain over the record: {stage1['gain']:+.4f} +- {stage1['ci95']:.4f} bpb, p = {stage1['p_beyond_margin']:.2g} "
              f"(beyond a margin of {RECORD_MARGIN})")
        if stats.mean_sd(ok)[1] < LOW_VARIANCE_RATIO * stats.mean_sd(incumbent)[1]:
            verdict["flag"] = "seed SD is far below the incumbent's; read the entry for a fixed init or data order"
            print(f"  flag: {verdict['flag']}")
        if stage1["passes"]:
            print("  passed; confirming on new seeds", flush=True)
            second = run_batch(entry, fresh_seeds(len(seeds)), args.split, track, logs, jobs, label="  (confirmation)")
            log["confirmation"] = second
            good = [r["bpb"] for r in second if r["status"] == "ok"]
            if len(good) == len(second):
                stage2 = judge(good, incumbent)
                verdict["stage2"] = stage2
                verdict["record"] = stage2["passes"]
                print(f"  confirmation: {stage2['gain']:+.4f} +- {stage2['ci95']:.4f} bpb, p = {stage2['p_beyond_margin']:.2g}")
        print("  NEW RECORD" if verdict["record"] else "  not a record")
        log["verdict"] = verdict
    log["wall_seconds"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log, indent=1))
    print(f"  log: {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    if len(ok) != len(results):
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child-train":
        child_train(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif len(sys.argv) > 1 and sys.argv[1] == "--child-eval":
        child_eval(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        main()
