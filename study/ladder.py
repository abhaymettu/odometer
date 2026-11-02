"""Seat a new record. Every record passes the same gate, whoever wrote it: the record rule, then a fresh written review.

    python -m study.ladder 01-gpt entries/gpt/train.py --review reviews/gpt.md --source "recipe: Karpathy nanoGPT"
    python -m study.ladder 02-x study/attempts/a/claims/1/train.py --review r.md --verdict study/attempts/a/claims/1/verify.json --source "agent: a/1"

The incumbent's numbers come from 8 fresh test seeds run here and never used for any decision. A challenger's own
confirmation seeds are not reused, because they were selected for passing.
"""

import argparse
import hashlib
import pathlib
import shutil
import time

import verify
from study import rules
from study.common import ROOT, layout, read_json, run_verify, write_json


def seat_problems(current, verdict_log, against_name, entry_sha, review_text):
    """Why this entry may not be seated, or an empty list. The first record has nothing to beat, so no verdict."""
    problems = [] if review_text.strip() else ["a record needs a written review"]
    if current is None:
        return problems
    if verdict_log is None:
        return problems + [f"--verdict is required: {current['name']} is the record to beat"]
    if not verdict_log.get("verdict", {}).get("record"):
        problems.append("that log's verdict is not a record")
    if verdict_log.get("sha256") != entry_sha:
        problems.append("that log scored a different file")
    if against_name != current["name"]:
        problems.append(f"that log beat {against_name}, but the record is now {current['name']}")
    return problems


def measure(entry, tmp, split, smoke, jobs):
    code, stdout = run_verify(entry, tmp / f"{split}.json", split, rules.N_SEEDS, smoke, jobs)
    log = read_json(tmp / f"{split}.json") if code == 0 else None
    if not log or "summary" not in log:
        raise SystemExit(f"the entry does not score on {split}:\n{stdout}")
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("entry", type=pathlib.Path)
    ap.add_argument("--review", type=pathlib.Path, required=True)
    ap.add_argument("--verdict", type=pathlib.Path, help="the verify.py --against log that passed the record rule")
    ap.add_argument("--source", default="unstated")
    ap.add_argument("--smoke", action="store_true", help="the smoke tree and verify.py --smoke; the numbers are not results")
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()

    place = layout(args.smoke)
    rec = place.records / args.name
    if rec.exists():
        raise SystemExit(f"{rec} already exists")
    current = read_json(place.current) if place.current.exists() else None
    verdict_log = read_json(args.verdict) if args.verdict else None
    against_name = read_json(verdict_log["verdict"]["against"])["name"] if verdict_log and "verdict" in verdict_log else None
    entry_sha = hashlib.sha256(args.entry.read_bytes()).hexdigest()
    problems = seat_problems(current, verdict_log, against_name, entry_sha, args.review.read_text(encoding="utf-8"))
    if problems:
        raise SystemExit("not seated:\n  " + "\n  ".join(problems))

    tmp = rec.with_name(rec.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copy2(args.entry, tmp / "train.py")
    shutil.copy2(args.review, tmp / "review.md")
    dev = measure(tmp / "train.py", tmp, "dev", args.smoke, args.jobs)
    test = measure(tmp / "train.py", tmp, "test", args.smoke, args.jobs)
    log = {"name": args.name, "dir": str(rec.relative_to(ROOT)), "source": args.source, "smoke": args.smoke, "split": "test",
           "seated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sha256": entry_sha, "code_sha256": verify.code_hash(),
           "track_version": verify.TRACK["version"], "dev": {**dev["summary"], "seeds": dev["seeds"]},
           "confirmation": test["seeds"], "seeds": test["seeds"], "confirmation_summary": test["summary"],
           "verdict": verdict_log.get("verdict") if verdict_log else None, "replaces": current["name"] if current else None}
    write_json(tmp / "log.json", log)
    tmp.rename(rec)
    write_json(place.current, log)
    print(f"{args.name} seated: dev {dev['summary']['mean_bpb']:.4f} sd {dev['summary']['sd']:.4f}, "
          f"test {test['summary']['mean_bpb']:.4f} sd {test['summary']['sd']:.4f}")


if __name__ == "__main__":
    main()
