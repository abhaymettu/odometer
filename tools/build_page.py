"""Build the page's data and the README's generated sections from the logs in this repo, so no number in them is typed by hand.

    python -m study.analyze && python tools/figures.py && python tools/build_page.py

Prose lives in docs/copy.json, with {placeholders} for every number. Record names and notes live in records/names.json.
"""

import json
import pathlib
import re
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
REPO = "https://github.com/abhaymettu/odometer"


def records():
    current = json.loads((ROOT / "records/current.json").read_text())["name"]
    names = json.loads((ROOT / "records/names.json").read_text())
    rows = []
    for d in sorted((ROOT / "records").glob("[0-9]*"), reverse=True):
        log = json.loads((d / "log.json").read_text())
        out = DOCS / "records" / d.name
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(d / "train.py", out / "train.py")
        shutil.copy2(d / "review.md", out / "review.md")
        rows.append({"name": d.name, "title": names[d.name]["title"], "who": names[d.name]["who"], "why": names[d.name].get("why", ""),
                     "source": log["source"], "test": log["confirmation_summary"]["mean_bpb"], "sd": log["confirmation_summary"]["sd"],
                     "dev": log["dev"]["mean_bpb"], "seeds": [s["bpb"] for s in log["confirmation"]],
                     "current": d.name == current, "code": f"records/{d.name}/train.py", "review": f"records/{d.name}/review.md",
                     "links": [["Code on GitHub", f"{REPO}/blob/main/records/{d.name}/train.py"],
                               ["Seed logs", f"{REPO}/tree/main/records/{d.name}"]]})
    return rows


def series_values():
    r = json.loads((ROOT / "study/results.json").read_text())
    by_claim, by_attempt = r["survival_per_claim"], r["survival_per_attempt"]
    costs = [json.loads(p.read_text()).get("final_result", {}).get("total_cost_usd") for p in sorted((ROOT / "study/attempts").glob("attempt-*/attempt.json"))]
    costs = [c for c in costs if c is not None]
    nulls = r["null_control"]
    return {"attempts": str(by_attempt["n"]), "claims": str(by_claim["n"]), "survived": str(by_claim["k"]),
            "claim_lo": f"{by_claim['lo']:.2f}", "claim_hi": f"{by_claim['hi']:.2f}",
            "attempt_survived": str(by_attempt["k"]), "attempt_lo": f"{by_attempt['lo']:.2f}", "attempt_hi": f"{by_attempt['hi']:.2f}",
            "records": str(r["records"]), "shrink": f"{r['shrinkage']['median']:.2f}", "best_gain": f"{r['best_verified_gain']:.4f}",
            "seed_runs_lo": str(r["seed_runs_per_claim"]["min"]), "seed_runs_hi": str(r["seed_runs_per_claim"]["max"]),
            "cost_lo": f"{min(costs):.2f}", "cost_hi": f"{max(costs):.2f}", "null_records": str(len(nulls)),
            "null_false": str(sum(n["false_replicated"]["k"] for n in nulls.values())), "null_n": str(sum(n["nulls"] for n in nulls.values())),
            "null_refuted": str(sum(n["false_refuted"] for n in nulls.values()))}, r["claims_table"]


def fair_values():
    path = ROOT / "study/fairness/result.json"
    if not path.exists():
        return dict.fromkeys(("fair_machines", "fair_max", "fair_ratio", "fair_seeds", "tolerance", "fair_p", "fair_entry", "fair_reference"), "?"), []
    f = json.loads(path.read_text())
    rows = f["rows"]
    biggest = max(x["max_abs_diff"] for x in rows)
    return {"fair_machines": str(len(rows) + 1), "fair_max": f"{biggest:.1e}", "fair_ratio": f"{0.002 / biggest:,.0f}", "fair_seeds": str(rows[0]["seeds"]),
            "tolerance": f"{f['tolerance']:g}", "fair_p": f"{max(x['tost_p'] for x in rows):.1e}", "fair_entry": f["entry"], "fair_reference": f["reference"]}, rows


def quick_values():
    path = ROOT / "study/fairness/quickstart.json"
    if not path.exists():
        return {"quick_seconds": "?", "quick_run": REPO}
    q = json.loads(path.read_text())
    return {"quick_seconds": str(q["seconds"]), "quick_run": q["run"]}


def fill(text, values):
    return re.sub(r"\{(\w+)\}", lambda m: values[m.group(1)], text)


def claims_rows(table):
    return [(t["attempt"].removeprefix("attempt-"), t["base"], f"{t['claimed_gain']:.4f}", f"{t['gain']:.4f} ± {t['ci95']:.4f}", t["outcome"], "yes" if t["record"] else "no") for t in table]


CLAIMS_HEAD = ("Attempt", "Started from", "Claimed gain", "Verified gain, 95%", "Outcome", "Record")
FAIR_HEAD = ("Machine", "Same seeds", "Ticks identical", "Largest difference, bits per byte", "Paired TOST p")


def fair_rows(rows):
    return [(x["machine"], str(x["seeds"]), "yes" if x["same_ticks"] else "NO", f"{x['max_abs_diff']:.1e}", f"{x['tost_p']:.1e}") for x in rows]


def html_table(head, rows):
    return ('<table class="plain"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table>")


def md_table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head), *("| " + " | ".join(r) + " |" for r in rows)])


def records_md(rows):
    body = [(str(len(rows) - i), f"[{r['title']}](records/{r['name']})", f"{r['test']:.4f} ± {r['sd']:.4f}", r["who"]) for i, r in enumerate(rows)]
    return md_table(("", "Record", "Test bits per byte", "From"), body)


def put(text, key, body):
    return re.sub(rf"<!-- {key} -->.*?<!-- /{key} -->", lambda m: f"<!-- {key} -->\n{body}\n<!-- /{key} -->", text, flags=re.S)


def tex_table(head, rows, spec):
    esc = lambda c: str(c).replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$")
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(map(esc, head)) + r" \\", r"\midrule"]
    return "\n".join(lines + [" & ".join(map(esc, r)) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def paper(values, claims, fair_table):
    """Everything numeric in the paper comes through paper/numbers.tex and the two generated tables."""
    out = ROOT / "paper"
    macro = lambda key: "v" + key.replace("_", "")
    (out / "numbers.tex").write_text("".join(rf"\newcommand{{\{macro(k)}}}{{{v.replace('_', chr(92) + '_').replace('%', chr(92) + '%')}}}" + "\n" for k, v in values.items()))
    (out / "claims.tex").write_text(tex_table(CLAIMS_HEAD, claims_rows(claims), "lllllc"))
    (out / "fair.tex").write_text(tex_table(FAIR_HEAD, fair_rows(fair_table), "lcccc"))


def main():
    rows = records()
    head = rows[0]
    values, claims = series_values()
    fair, fair_table = fair_values()
    values |= fair | quick_values()
    values |= {"attacks": str(sum(1 for p in (ROOT / "tests/attacks").glob("*.py") if p.stem != "legit_tiny")), "record_test": f"{head['test']:.4f}", "record_name": head["title"],
               "agent_records": str(sum(r["source"].startswith("agent") for r in rows))}
    tables = {"claims_table": html_table(CLAIMS_HEAD, claims_rows(claims)), "fair_table": html_table(FAIR_HEAD, fair_rows(fair_table))}
    copy = json.loads((DOCS / "copy.json").read_text())
    data = {"records": rows, "lead": fill(copy["lead"], values), "html": {k: fill(v, values | tables) for k, v in copy["html"].items()}}
    (DOCS / "data.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    for name in ("__init__.py", "_engine.py", "_data.py"):
        shutil.copy2(ROOT / "odometer" / name, DOCS / "demo" / "odometer" / name)
    shutil.copy2(ROOT / head["code"], DOCS / "demo" / "train.py")

    tables = {"claims_table": md_table(CLAIMS_HEAD, claims_rows(claims)), "fair_table": md_table(FAIR_HEAD, fair_rows(fair_table))}
    readme = (ROOT / "README.md").read_text()
    readme = put(readme, "records", records_md(rows))
    for key, body in copy["readme"].items():
        readme = put(readme, key, fill(body, values | tables))
    (ROOT / "README.md").write_text(readme)
    (ROOT / "site/writeup.md").write_text(fill((ROOT / "site/writeup.in.md").read_text(), values))
    paper(values, claims, fair_table)
    print(f"{len(rows)} records, current {head['name']} {head['test']:.4f}")


if __name__ == "__main__":
    main()
