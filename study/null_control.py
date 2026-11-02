"""The current record resubmitted unchanged, through the same scoring as a claim, stage 1 only.

    python -m study.null_control

The nulls claim no gain. A `replicated` here is a false positive of the classification rule; `stage1` is what the record
rule's first stage would have said. Done nulls are kept, so a rerun only does the missing ones.
"""

import argparse

import verify
from study import rules, scrub
from study.common import ROOT, layout, read_json, write_json
from study.verify_claims import check_base, score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="verify.py --smoke, in the smoke tree; the numbers are not results")
    ap.add_argument("--nulls", type=int, default=rules.N_NULLS, help="smoke runs only; the real count is preregistered")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--record", help="the record's folder name, when it is no longer the current one")
    args = ap.parse_args()
    if args.nulls != rules.N_NULLS and not args.smoke:
        raise SystemExit(f"a real null control runs {rules.N_NULLS} nulls")
    place = layout(args.smoke)
    log = place.records / args.record / "log.json" if args.record else place.current
    base = read_json(log)
    check_base(base, args.smoke)
    out = place.nulls / base["name"]
    incumbent = [r["bpb"] for r in base["confirmation"]]
    for i in range(1, args.nulls + 1):
        dest = out / str(i)
        if (dest / "outcome.json").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        row = score(ROOT / base["dir"] / "train.py", 0.0, base, log, dest, args.smoke, args.jobs, stage1_only=True)
        row["stage1"] = verify.judge(row["bpbs"], incumbent) if "bpbs" in row else None
        write_json(dest / "outcome.json", row)
        print(f"  null {i}: {row['outcome']}" + (f", gain {row['gain']:+.4f}" if "gain" in row else ""), flush=True)
    scrub.check_tree(out, scrub.env_literals())


if __name__ == "__main__":
    main()
