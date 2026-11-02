"""Run the demo natively and store what the page compares a browser's run with.

    python tools/demo_reference.py "M4 Pro, native"

Rows from other machines come from the fairness workflow, in study/fairness/demo/<machine>.json. Only rows of the record
now in docs/demo count, and any whose reading differs from this machine's stops the build, since ticks must match exactly.
"""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEMO = ROOT / "docs" / "demo"


def main():
    report = json.loads(subprocess.run([sys.executable, str(DEMO / "harness.py")], check=True, capture_output=True, text=True).stdout)
    rows = [{"machine": sys.argv[1], "train_seconds": report["train_seconds"]}]
    for path in sorted((ROOT / "study/fairness/demo").glob("*.json")):
        other = json.loads(path.read_text())
        if other["text"] != report["text"] or other["reading"] != report["reading"]:
            raise SystemExit(f"{path.name}: reading {other['reading']} against {report['reading']}, or different text")
        rows.append({"machine": path.stem, "train_seconds": other["train_seconds"]})
    out = {"budget": report["budget"], "reading": report["reading"], "bpb": report["bpb"], "rows": rows}
    (DEMO / "reference.json").write_text(json.dumps(out) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
