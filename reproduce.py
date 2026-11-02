"""Rerun a headline number with one command. Nothing here is trusted: each path calls the public verifier or reads its logs.

    python reproduce.py            the current record on 2 fresh test seeds, against its stored mean
    python reproduce.py table      every record on 8 fresh test seeds
    python reproduce.py attacks    the adversarial entries, all of which must be rejected or priced out
    python reproduce.py series     the agent series' numbers, from the stored claim logs
"""

import json
import pathlib
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent


def records():
    return sorted(p for p in (ROOT / "records").iterdir() if (p / "log.json").exists())


def verify(record, seeds):
    stored = json.loads((record / "log.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "log.json"
        code = subprocess.run([sys.executable, str(ROOT / "verify.py"), str(record / "train.py"), "--seeds", str(seeds), "--split", "test",
                               "--out", str(out)], cwd=ROOT).returncode
        if code:
            raise SystemExit(f"{record.name} did not score (verify.py exit {code})")
        scores = [s["bpb"] for s in json.loads(out.read_text(encoding="utf-8"))["seeds"]]
    mine, kept, sd = sum(scores) / len(scores), stored["confirmation_summary"]["mean_bpb"], stored["confirmation_summary"]["sd"]
    print(f"{record.name}: {mine:.4f} bpb on {seeds} fresh seeds; stored {kept:.4f}, seed SD {sd:.4f}, off by {mine - kept:+.4f}")


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "record"
    if not (ROOT / "data" / "train.bin").exists() and what in ("record", "table"):
        subprocess.run([sys.executable, str(ROOT / "prepare.py")], check=True)
    start = time.time()
    if what == "record":
        current = json.loads((ROOT / "records" / "current.json").read_text(encoding="utf-8"))
        verify(ROOT / current["dir"], 2)
    elif what == "table":
        for record in records():
            verify(record, 8)
    elif what == "attacks":
        raise SystemExit(subprocess.run([sys.executable, "-m", "pytest", "tests/test_attacks.py", "-q"], cwd=ROOT).returncode)
    elif what == "series":
        raise SystemExit(subprocess.run([sys.executable, "-m", "study.analyze"], cwd=ROOT).returncode)
    else:
        raise SystemExit(__doc__)
    print(f"done in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
