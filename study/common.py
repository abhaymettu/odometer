"""Where the study keeps things, and the one way it calls the verifier."""

import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parent.parent
ATT_ROOT = pathlib.Path.home() / "att"


def layout(smoke=False):
    """Smoke runs get their own tree, so their numbers can never reach the results."""
    base = ROOT / "study" / "smoke" if smoke else ROOT / "study"
    records = (base if smoke else ROOT) / "records"
    return SimpleNamespace(smoke=smoke, base=base, attempts=base / "attempts", nulls=base / "nulls", records=records,
                           current=records / "current.json")


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_verify(entry, out, split, seeds, smoke, jobs, against=None):
    """The pristine verify.py in this repo, on its own command line. Returns (exit code, stdout)."""
    cmd = [sys.executable, "verify.py", str(entry), "--seeds", str(seeds), "--split", split, "--out", str(out), "--jobs", str(jobs)]
    cmd += ["--smoke"] if smoke else []
    cmd += ["--against", str(against)] if against else []
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    return proc.returncode, proc.stdout
