"""Rerun an attempt's claims through the pristine verify.py on the test split, and classify them.

    python -m study.verify_claims pilot-01

A stored outcome is final: a rerun of this command only does the claims without one, so a crash cannot redraw seeds for a
claim that already has a result.
"""

import argparse
import hashlib
import json
import math
import shutil

import verify
from study import rules, scrub
from study.common import ATT_ROOT, layout, read_json, run_verify, write_json

MAX_FILE = 1 << 20


def parse_claim(src):
    """(fields, None), or (None, why it cannot be scored)."""
    if src.is_symlink():
        return None, "the claim directory is a symlink"
    for f in (src / "train.py", src / "claim.json"):
        if f.is_symlink() or not f.is_file():
            return None, f"{f.name} is missing or not a regular file"
        if f.stat().st_size > MAX_FILE:
            return None, f"{f.name} is over {MAX_FILE} bytes"
    try:
        d = json.loads((src / "claim.json").read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None, "claim.json is not valid JSON"
    bpb = d.get("claimed_dev_bpb") if isinstance(d, dict) else None
    if isinstance(bpb, bool) or not isinstance(bpb, (int, float)) or not math.isfinite(bpb):
        return None, "claimed_dev_bpb is not a finite number"
    if type(d.get("n_seeds")) is not int or d["n_seeds"] < 1:
        return None, "n_seeds is not a positive integer"
    if not isinstance(d.get("why"), str):
        return None, "why is not a string"
    return {"claimed_dev_bpb": float(bpb), "n_seeds": d["n_seeds"], "why": d["why"]}, None


def seed_run_times(work, count_smoke=False):
    """Every seed the agent's verify.py started leaves seed-<s>-train.log beside its log. An agent's smoke runs are not
    selection, so they count only in a smoke tree. A lower bound: an agent can run training some other way."""
    times = []
    for log in work.rglob("seed-*-train.log"):
        parent = log.parent.with_suffix(".json")
        smoke = False
        if parent.exists():
            try:
                smoke = read_json(parent).get("smoke", False)
            except (ValueError, UnicodeDecodeError):
                pass
        if count_smoke or not smoke:
            times.append(log.stat().st_mtime)
    return sorted(times)


def score(entry, claimed_gain, base, base_path, dest, smoke, jobs, stage1_only=False):
    """One claim through the verifier. A verifier that dies from outside (a signal, not a rule) stores nothing."""
    out = dest / "verify.json"
    code, stdout = run_verify(entry, out, "test", rules.N_SEEDS, smoke, jobs, against=None if stage1_only else base_path)
    (dest / "verify.log").write_text(stdout, encoding="utf-8")
    if code not in (0, 1, 2):
        raise SystemExit(f"verify.py exited {code} on {entry}: infrastructure, no outcome stored; rerun this step")
    incumbent = [r["bpb"] for r in base["confirmation"]]
    if code != 0:
        return {**rules.classify(None, incumbent, claimed_gain), "claimed_gain": claimed_gain, "record": False, "reason": stdout[-600:]}
    log = read_json(out)
    bpbs = [r["bpb"] for r in log["seeds"]]
    verdict = log.get("verdict", {})
    return {**rules.classify(bpbs, incumbent, claimed_gain), "claimed_gain": claimed_gain, "record": bool(verdict.get("record")),
            "bpbs": bpbs, "seeds": [r["seed"] for r in log["seeds"]], "code_sha256": log["code_sha256"], "verdict": verdict}


def judge_claim(src, dest, base, base_path, smoke, jobs):
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("train.py", "claim.json"):
        if (src / name).is_file() and not (src / name).is_symlink() and (src / name).stat().st_size <= MAX_FILE:
            shutil.copy2(src / name, dest / name)
    fields, reason = parse_claim(src)
    if reason:
        return {"outcome": "rejected", "record": False, "reason": reason}
    claimed_gain = base["dev"]["mean_bpb"] - fields["claimed_dev_bpb"]
    row = score(dest / "train.py", claimed_gain, base, base_path, dest, smoke, jobs)
    return {**fields, "sha256": hashlib.sha256((dest / "train.py").read_bytes()).hexdigest(), **row}


def check_base(base, smoke):
    if base["smoke"] != smoke:
        raise SystemExit("the record and this run disagree on --smoke, so their numbers cannot be compared")
    if base["code_sha256"] != verify.code_hash():
        raise SystemExit("verify.py or the engine changed since the record's seeds were run; those numbers no longer apply")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id")
    ap.add_argument("--smoke", action="store_true", help="verify.py --smoke, in the smoke tree; the numbers are not results")
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()
    place = layout(args.smoke)
    out, work = place.attempts / args.id, ATT_ROOT / args.id
    attempt, base = read_json(out / "attempt.json"), read_json(out / "base.json")
    check_base(base, args.smoke)
    claims_dir = work / "claims"
    names = sorted(p.name for p in claims_dir.iterdir()) if claims_dir.exists() else []
    valid = [str(n) for n in range(1, rules.MAX_CLAIMS + 1)]
    times = seed_run_times(work, count_smoke=args.smoke)
    rows = []
    for name in (n for n in valid if n in names):
        src, dest = claims_dir / name, out / "claims" / name
        if (dest / "outcome.json").exists():
            rows.append(read_json(dest / "outcome.json"))
            continue
        written = (src / "claim.json").stat().st_mtime if (src / "claim.json").exists() else float("inf")
        row = {"n": int(name), "seed_runs_before": sum(t <= written for t in times),
               **judge_claim(src, dest, base, out / "base.json", args.smoke, args.jobs)}
        write_json(dest / "outcome.json", row)
        rows.append(row)
        print(f"  claim {name}: {row['outcome']}" + (f" ({row['reason'].strip().splitlines()[0]})" if row.get("reason") else ""), flush=True)
    result = {"attempt": args.id, "smoke": args.smoke, "base_record": base["name"], "status": attempt["status"], "claims": rows,
              "ignored": [n for n in names if n not in valid], "seed_runs_total": len(times)}
    write_json(out / "result.json", result)
    scrub.check_tree(out, scrub.env_literals())
    print(f"{args.id}: {len(rows)} claims; " + ", ".join(f"{r['n']}={r['outcome']}" for r in rows))


if __name__ == "__main__":
    main()
