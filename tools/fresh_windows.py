"""Do the gains hold on test windows no score has used? Every verifier score reads the same 1024 windows, so its plus-minus covers seed noise only.

    python tools/fresh_windows.py records/11-mixer-thin/train.py --seeds 7001,7002,7003,7004 --out out.json

Each seed trains through the verifier's own child, then one trained model is scored on the usual 1024 test windows and on the next 1024 of the
same permutation, which share no window with them. This decides nothing. Eval here has no twins and no eval cap, since it only reads records that passed both.
"""

import argparse
import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import verify  # noqa: E402

COUNT = verify.TRACK["eval_windows"]


def windows():
    """The usual test windows and the next ones along, cut the way odometer._data.heldout_windows cuts them."""
    import numpy as np

    from odometer import _data

    context = verify.TRACK["eval_context"]
    a = np.fromfile(_data.DATA / "heldout.bin", dtype=np.uint8)
    n = a.size // (context + 1)
    half = np.random.default_rng(20261101).permutation(n)[_data.HELDOUT_SPLITS["test"] :: 2]
    cut = a[: n * (context + 1)].reshape(n, context + 1)
    used, fresh = cut[np.sort(half[:COUNT])], cut[np.sort(half[COUNT : 2 * COUNT])]
    assert (used == _data.heldout_windows("test", context, COUNT)).all() and not set(half[:COUNT].tolist()) & set(half[COUNT : 2 * COUNT].tolist())
    return used, fresh


def child(entry, seed, tmp):
    import numpy as np

    import odometer as od
    from odometer import _engine

    tmp = pathlib.Path(tmp)
    track = json.loads((tmp / "track.json").read_text())
    context, batch = track["eval_context"], track["eval_batch"]
    _engine._M.reset(None, seed, mode="eval")
    module = verify.load_entry(entry, track)
    names = json.loads((tmp / "names.json").read_text())
    stored = np.load(tmp / "params.npz")
    arrays = {name: stored[f"p{i}"] for i, name in enumerate(names)}

    def score(rows):
        nll = 0.0
        for start in range(0, len(rows), batch):
            r = rows[start : start + batch]
            z = module.forward({k: od.Tensor(a) for k, a in arrays.items()}, od.Tensor(r[:, :context].astype(np.int32)))._d.astype(np.float64)
            lp = z - z.max(-1, keepdims=True)
            lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
            nll -= lp[np.arange(len(r))[:, None], np.arange(context), r[:, 1:]].sum()
        return float(nll / (len(rows) * context) / np.log(2))

    used, fresh = windows()
    (tmp / "scores.json").write_text(json.dumps({"used": score(used), "fresh": score(fresh)}))


def one(entry, seed):
    env = {"PATH": os.environ.get("PATH", ""), **{v: "1" for v in verify.THREAD_VARS}}
    with tempfile.TemporaryDirectory() as tmp:
        (pathlib.Path(tmp) / "track.json").write_text(json.dumps(verify.TRACK))
        subprocess.run([sys.executable, "-I", str(ROOT / "verify.py"), "--child-train", entry, str(seed), tmp], check=True, env=env, cwd=ROOT)
        subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), "--child", entry, str(seed), tmp], check=True, env=env, cwd=ROOT)
        train = json.loads((pathlib.Path(tmp) / "train.json").read_text())
        return {"seed": seed, "status": train["status"], "train_ticks": train["train_ticks"], **json.loads((pathlib.Path(tmp) / "scores.json").read_text())}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return child(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entry")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(lambda s: one(args.entry, s), seeds))
    pathlib.Path(args.out).write_text(json.dumps({"entry": args.entry, "windows": COUNT, "seeds": rows}, indent=1) + "\n")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
