"""Regenerate every study number from the attempts folder: study/results.md and study/results.json.

    python -m study.analyze
"""

import argparse
import itertools
import json
import statistics
import sys

import stats
from study import rules
from study.common import layout, read_json

OUTCOMES = ("replicated", "refuted", "inconclusive", "rejected")


def rate(k, n):
    lo, hi = stats.wilson(k, n) if n else (None, None)
    return {"k": k, "n": n, "rate": k / n if n else None, "lo": max(0.0, lo) if n else None, "hi": min(1.0, hi) if n else None}


def spread(xs):
    return {"n": len(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs)} if xs else {"n": 0, "median": None, "min": None, "max": None}


def summarize(rows):
    """rows: [{"id", "attempt", "result"}], with result None until verify_claims has run. Pure: the tests call this directly."""
    rows = [r for r in rows if not r["attempt"] or r["attempt"]["status"] != "probe_only"]
    pending = [r["id"] for r in rows if not r["attempt"] or r["result"] is None]
    done = [r for r in rows if r["attempt"] and r["result"] is not None]
    kept = [r for r in done if rules.counted(r["attempt"]["status"], r["result"]["claims"])]
    claims = [{"attempt": r["id"], "base": r["result"]["base_record"], **c} for r in kept for c in r["result"]["claims"]]
    scored = [c for c in claims if c["outcome"] != "rejected"]
    positive = [c for c in scored if c["claimed_gain"] > 0]
    by_base = {}
    for c in claims:
        by_base.setdefault(c["base"], []).append(c["outcome"] == "replicated")
    return {
        "attempts": {"started": len(rows), "counted": len(kept), "excluded": [r["id"] for r in done if r not in kept], "pending": pending,
                     "with_no_claims": sum(not r["result"]["claims"] for r in kept),
                     "with_tool_denials": sum(bool((r["attempt"].get("final_result") or {}).get("permission_denials")) for r in kept)},
        "claims": {"total": len(claims), **{o: sum(c["outcome"] == o for c in claims) for o in OUTCOMES}},
        "survival_per_claim": rate(sum(c["outcome"] == "replicated" for c in claims), len(claims)),
        "survival_per_attempt": rate(sum(any(c["outcome"] == "replicated" for c in r["result"]["claims"]) for r in kept), len(kept)),
        "shrinkage": {**spread([c["gain"] / c["claimed_gain"] for c in positive]), "left_out_no_claimed_gain": len(scored) - len(positive)},
        "records": sum(bool(c["record"]) for c in claims),
        "seed_runs_per_claim": spread([c["seed_runs_before"] for c in claims]),
        "best_verified_gain": max((c["gain"] for c in scored), default=None),
        "survival_by_base_record": {b: rate(sum(xs), len(xs)) for b, xs in sorted(by_base.items())},
        "claims_table": [{k: c.get(k) for k in ("attempt", "n", "base", "claimed_gain", "gain", "ci95", "p", "outcome", "record", "seed_runs_before")} for c in claims],
    }


def null_summary(outcomes, stored):
    """False positives among nulls, plus two things 8 nulls against one stored batch cannot show alone: every batch
    against every other, and how far the stored batch sits from the record's own long-run mean."""
    batches = [o["bpbs"] for o in outcomes if "bpbs" in o]
    pairs = [stats.welch(a, b)["p_one_sided"] < rules.ALPHA for a, b in itertools.permutations(batches, 2)]
    summary = {"nulls": len(outcomes), "false_replicated": rate(sum(o["outcome"] == "replicated" for o in outcomes), len(outcomes)),
               "false_refuted": sum(o["outcome"] == "refuted" for o in outcomes), "rejected": sum(o["outcome"] == "rejected" for o in outcomes),
               "stage1_passes": sum(bool(o.get("stage1") and o["stage1"]["passes"]) for o in outcomes),
               "pairwise_false_positives": rate(sum(pairs), len(pairs)), "stored_offset_in_se": None}
    if batches:
        pooled = stats.mean_sd([b for batch in batches for b in batch])[0]
        mean, sd = stats.mean_sd(stored)
        summary["stored_offset_in_se"] = (mean - pooled) / (sd / len(stored) ** 0.5) if sd else None
    return summary


def fmt(x, digits=4):
    return "n/a" if x is None else f"{x:+.{digits}f}" if isinstance(x, float) else str(x)


def plain(x):
    return "n/a" if x is None else f"{x:g}"


def rate_cell(r):
    return f"{r['k']}/{r['n']} (Wilson 95% {r['lo']:.3f} to {r['hi']:.3f})" if r["n"] else "0/0 (no data)"


def markdown(s, smoke):
    a, c = s["attempts"], s["claims"]
    lines = ["# odometer agent series: results", ""]
    lines += ["**SMOKE RUN. These numbers are not results.**", ""] if smoke else []
    lines += ["| Measure | Value |", "|---|---|",
              f"| Attempts counted | {a['counted']} of {a['started']} started |",
              f"| Attempts excluded (infrastructure, no claim) | {len(a['excluded'])} |",
              f"| Attempts waiting for verify_claims | {len(a['pending'])} |",
              f"| Counted attempts with no claim | {a['with_no_claims']} |",
              f"| Counted attempts with a denied tool | {a['with_tool_denials']} |",
              f"| Claims | {c['total']}: {c['replicated']} replicated, {c['refuted']} refuted, {c['inconclusive']} inconclusive, {c['rejected']} rejected |",
              f"| Survival per claim (primary) | {rate_cell(s['survival_per_claim'])} |",
              f"| Survival per attempt (primary) | {rate_cell(s['survival_per_attempt'])} |",
              f"| Median shrinkage g/c, claims with c > 0 | {fmt(s['shrinkage']['median'], 3)} (n={s['shrinkage']['n']}, {s['shrinkage']['left_out_no_claimed_gain']} claimed no gain) |",
              f"| Records | {s['records']} |",
              f"| Seed-runs before a claim, median (range) | {plain(s['seed_runs_per_claim']['median'])} ({plain(s['seed_runs_per_claim']['min'])} to {plain(s['seed_runs_per_claim']['max'])}) |",
              f"| Best verified gain, bpb | {fmt(s['best_verified_gain'])} |"]
    for base, r in s["survival_by_base_record"].items():
        lines.append(f"| Survival from record {base} | {rate_cell(r)} |")
    for name, n in s["null_control"].items():
        lines.append(f"| Null control on {name} | {n['nulls']} nulls: false replicated {rate_cell(n['false_replicated'])}; false refuted {n['false_refuted']}; "
                     f"stage 1 passes {n['stage1_passes']}; batch against batch {rate_cell(n['pairwise_false_positives'])}; "
                     f"stored batch offset {fmt(n['stored_offset_in_se'], 2)} SE |")
    lines += ["", "| Attempt | Claim | Base | Claimed gain | Verified gain | 95% half-width | p | Outcome | Record | Seed-runs before |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in s["claims_table"]:
        lines.append(f"| {t['attempt']} | {t['n']} | {t['base']} | {fmt(t['claimed_gain'])} | {fmt(t['gain'])} | {fmt(t['ci95'])} | {fmt(t['p'], 3)} | {t['outcome']} | {t['record']} | {t['seed_runs_before']} |")
    if s.get("pilots"):
        lines += ["", "Pilots ran before the series was registered and count toward nothing above.", ""]
        lines += [f"- {t['attempt']} claim {t['n']}: {t['base']}, claimed {fmt(t['claimed_gain'])}, verified {fmt(t['gain'])}, {t['outcome']}, record {t['record']}"
                  for t in s["pilots"]]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="read the smoke tree instead")
    args = ap.parse_args()
    place = layout(args.smoke)
    rows, pilots = [], []
    for d in sorted(p for p in place.attempts.glob("*") if p.is_dir()):
        attempt = read_json(d / "attempt.json") if (d / "attempt.json").exists() else None
        result = read_json(d / "result.json") if (d / "result.json").exists() else None
        if attempt and attempt["smoke"] != args.smoke:
            sys.exit(f"{d} is from a different kind of run than this tree: smoke={attempt['smoke']}")
        (pilots if d.name.startswith("pilot-") else rows).append({"id": d.name, "attempt": attempt, "result": result})
    summary = summarize(rows)
    summary["smoke"] = args.smoke
    summary["pilots"] = summarize(pilots)["claims_table"]
    summary["null_control"] = {}
    for d in sorted(p for p in place.nulls.glob("*") if p.is_dir()):
        outcomes = [read_json(p) for p in sorted(d.glob("*/outcome.json"))]
        stored = [r["bpb"] for r in read_json(place.records / d.name / "log.json")["confirmation"]]
        summary["null_control"][d.name] = null_summary(outcomes, stored)
    text = markdown(summary, args.smoke)
    place.base.mkdir(parents=True, exist_ok=True)
    (place.base / "results.md").write_text(text, encoding="utf-8")
    (place.base / "results.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
