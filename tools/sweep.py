"""What each tick buys: rerun entries at other budgets, recipes unchanged, and write one JSON row per run.

    python tools/sweep.py records/04-ppm/train.py records/05-mixer/train.py --budgets 2.5e9,1e10,4e10 --seeds 3 --out sweep.json

Recipes were tuned at the track's budget and are not retuned here, so a recipe's score at another budget is a floor on what
that recipe could do there. The scores are dev numbers from a track with one number changed, so none of them can hold a record.
"""

import argparse
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import verify  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entries", nargs="+")
    ap.add_argument("--budgets", default="2.5e9,1e10,4e10")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for entry in args.entries:
            for budget in (float(b) for b in args.budgets.split(",")):
                track = {**verify.TRACK, "train_budget": int(budget), "wall_limit_seconds": 3 * 3600}
                results = verify.run_batch(pathlib.Path(entry).resolve(), verify.fresh_seeds(args.seeds), "dev", track,
                                           pathlib.Path(tmp), args.jobs or 2, label=f"  {entry} {budget:g}")
                rows.append({"entry": entry, "budget": int(budget), "seeds": results})
                pathlib.Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
