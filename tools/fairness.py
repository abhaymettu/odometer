"""Does the same entry get the same score on every machine? Compare verify.py logs of the same seeds.

    python tools/fairness.py study/fairness/*.json --tolerance 1e-3

Ticks must match exactly, since they are a pure function of the program. Scores are floats that pass through each machine's
BLAS, so they may differ, and the test is paired: for each machine against the reference (the first log), a TOST on the
same-seed differences against +-tolerance. Exits 1 if any tick count differs or any machine fails the test.
"""

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import stats  # noqa: E402


def paired_tost(diffs, tolerance):
    """The larger of the two one-sided p values that the mean difference is inside +-tolerance. All-zero differences pass."""
    m, sd = stats.mean_sd(diffs)
    if sd == 0:
        return 0.0 if abs(m) < tolerance else 1.0
    se, df = sd / math.sqrt(len(diffs)), len(diffs) - 1
    return max(1 - stats.t_cdf((m + tolerance) / se, df), stats.t_cdf((m - tolerance) / se, df))


def compare(reference, log, tolerance):
    base = {s["seed"]: s for s in reference["seeds"]}
    rows = [(base[s["seed"]], s) for s in log["seeds"] if s["seed"] in base]
    if len(rows) < 2:
        raise SystemExit(f"fewer than 2 shared seeds with {log['machine']['platform']}")
    diffs = [b["bpb"] - a["bpb"] for a, b in rows]
    same_ticks = all(a["train_ticks"] == b["train_ticks"] and a["eval_ticks_per_byte"] == b["eval_ticks_per_byte"] for a, b in rows)
    return {"seeds": len(rows), "max_abs_diff": max(abs(d) for d in diffs), "mean_diff": sum(diffs) / len(diffs),
            "tost_p": paired_tost(diffs, tolerance), "same_ticks": same_ticks}


def name_of(log):
    m = log["machine"]
    return f"{m['platform'].split('-')[0]} {m['machine']}, {m['cpus']} cores, {m['blas']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=pathlib.Path)
    ap.add_argument("--tolerance", type=float, default=1e-3, help="bits per byte, either way")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    logs = [json.loads(p.read_text(encoding="utf-8")) for p in args.logs]
    if len({log["code_sha256"] for log in logs}) != 1 or len({log["sha256"] for log in logs}) != 1:
        raise SystemExit("these logs ran different code or a different entry")
    reference = logs[0]
    print(f"entry {reference['entry']}, tolerance +-{args.tolerance:g} bpb, reference: {name_of(reference)}")
    rows, good = [], True
    print("| Machine | Same seeds | Ticks identical | Largest difference, bpb | TOST p |\n|---|---|---|---|---|")
    for log in logs[1:]:
        c = compare(reference, log, args.tolerance)
        good &= c["same_ticks"] and c["tost_p"] < args.alpha
        rows.append({"machine": name_of(log), **c})
        print(f"| {name_of(log)} | {c['seeds']} | {'yes' if c['same_ticks'] else 'NO'} | {c['max_abs_diff']:.2e} | {c['tost_p']:.2g} |")
    if args.out:
        args.out.write_text(json.dumps({"entry": reference["entry"], "tolerance": args.tolerance, "reference": name_of(reference), "rows": rows}, indent=1),
                            encoding="utf-8")
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
