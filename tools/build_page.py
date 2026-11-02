"""Build the page's data from the logs in this repo, so no number on it is typed by hand.

    python tools/build_page.py
"""

import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
REPO = "https://github.com/abhaymettu/odometer"

TITLES = {
    "01-gpt": ("Small GPT", "Recipe after Karpathy's nanoGPT, ported and tuned here"),
    "02-rnn": ("Character RNN", "Recipe after Karpathy's char-rnn, ported and tuned here"),
    "03-modern": ("Modern transformer", "Recipe from recent small-model practice, ported and tuned here"),
    "04-ppm": ("PPM with learned backoff", "Recipe after Cleary and Witten's PPM, ported and tuned here"),
    "05-mixer": ("Hashed context mixing", "Recipe after Mahoney's PAQ and Knoll's cmix, ported and tuned here"),
    "06-mixer-fold": ("Mixing, folded head, lazy tables", "Claude Opus 5.5, in a pilot before the series"),
}


def records():
    current = json.loads((ROOT / "records/current.json").read_text())["name"]
    rows = []
    for d in sorted((ROOT / "records").glob("[0-9]*"), reverse=True):
        log = json.loads((d / "log.json").read_text())
        title, who = TITLES.get(d.name, (d.name, log["source"]))
        out = DOCS / "records" / d.name
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(d / "train.py", out / "train.py")
        shutil.copy2(d / "review.md", out / "review.md")
        rows.append({"name": d.name, "title": title, "who": who, "source": log["source"], "test": log["confirmation_summary"]["mean_bpb"],
                     "sd": log["confirmation_summary"]["sd"], "dev": log["dev"]["mean_bpb"], "seeds": [s["bpb"] for s in log["confirmation"]],
                     "current": d.name == current, "code": f"records/{d.name}/train.py", "review": f"records/{d.name}/review.md",
                     "links": [["Code on GitHub", f"{REPO}/blob/main/records/{d.name}/train.py"],
                               ["Seed logs", f"{REPO}/tree/main/records/{d.name}"]]})
    return rows


def main():
    rows = records()
    head = rows[0]
    copy = json.loads((DOCS / "copy.json").read_text()) if (DOCS / "copy.json").exists() else {}
    data = {"records": rows, "lead": copy.get("lead", ""), "html": copy.get("html", {})}
    (DOCS / "data.json").write_text(json.dumps(data, indent=1) + "\n")
    for name in ("__init__.py", "_engine.py", "_data.py"):
        shutil.copy2(ROOT / "odometer" / name, DOCS / "demo" / "odometer" / name)
    shutil.copy2(ROOT / head["code"], DOCS / "demo" / "train.py")
    print(f"{len(rows)} records, current {head['name']} {head['test']:.4f}")


if __name__ == "__main__":
    main()
