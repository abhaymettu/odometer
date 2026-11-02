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
            "records": str(r["records"]), "shrink": f"{r['shrinkage']['median']:.3f}", "best_gain": f"{r['best_verified_gain']:.4f}",
            "shrink_lo": f"{r['shrinkage']['min']:.2f}", "shrink_hi": f"{r['shrinkage']['max']:.2f}",
            "shrink_below": str(sum(t["gain"] / t["claimed_gain"] < 1 for t in r["claims_table"] if t["claimed_gain"] > 0)),
            "gain_last": f"{[t['gain'] for t in r['claims_table'] if t['record']][-1]:.4f}",
            "seed_runs_lo": str(r["seed_runs_per_claim"]["min"]), "seed_runs_hi": str(r["seed_runs_per_claim"]["max"]),
            "cost_lo": f"{min(costs):.2f}", "cost_hi": f"{max(costs):.2f}", "null_records": str(len(nulls)),
            "null_false": str(sum(n["false_replicated"]["k"] for n in nulls.values())), "null_n": str(sum(n["nulls"] for n in nulls.values())),
            "null_refuted": str(sum(n["false_refuted"] for n in nulls.values()))}, r["claims_table"]


def twin_values():
    """The largest twin value at the track's budget, across every distinct seed run stored in the repo."""
    runs = {}

    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("twin_max_diff"), (int, float)) and 3.0e10 < (o.get("train_ticks") or 0) <= 4e10:
                runs[(o.get("seed"), o.get("bpb"))] = o["twin_max_diff"]
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for folder in ("records", "study/attempts", "study/nulls"):
        for path in (ROOT / folder).rglob("*.json"):
            walk(json.loads(path.read_text()))
    return {"twin_max": f"{max(runs.values()):.1e}", "twin_runs": str(len(runs))}


def seed_time_values():
    two, four = [], []
    for d in (ROOT / "records").glob("[0-9]*"):
        log = json.loads((d / "test.json").read_text())
        (two if log["machine"]["cpus"] == 2 else four).extend(s["train_seconds"] + s["eval_seconds"] for s in log["seeds"])
    return {"two_lo": f"{min(two):.0f}", "two_hi": f"{max(two):.0f}", "four_lo": f"{min(four):.0f}", "four_hi": f"{max(four):.0f}"}


def power_values():
    w = json.loads((ROOT / "study/power.json").read_text())
    rule = w["record_rule_power"]
    return {"p_half": f"{w['all_replicate_if_half']:.2f}", "rep_power": f"{w['replicated_power_one_sd']:.0%}", "rule_003": f"{rule['0.003']:.0%}", "rule_004": f"{rule['0.004']:.0%}"}


def fresh_values(rows):
    """Each record on the usual windows and on the next 1024, from tools/fresh_windows.py."""
    table, by = [], {}
    for r in reversed(rows):
        path = ROOT / "study/fresh_windows" / f"{r['name']}.json"
        if path.exists():
            seeds = json.loads(path.read_text())["seeds"]
            used, fresh = sum(s["used"] for s in seeds) / len(seeds), sum(s["fresh"] for s in seeds) / len(seeds)
            by[r["name"]] = (used, fresh)
            table.append((r["title"], f"{used:.4f}", f"{fresh:.4f}", f"{round(fresh - used, 4) + 0.0:+.4f}"))
    if len(by) < 2:
        return dict.fromkeys(("fresh_first", "fresh_used_gain", "fresh_fresh_gain", "fresh_last_used", "fresh_last_fresh"), "?"), table
    names = list(by)
    return {"fresh_first": names[0][:2], "fresh_used_gain": f"{by[names[0]][0] - by[names[-1]][0]:.4f}", "fresh_fresh_gain": f"{by[names[0]][1] - by[names[-1]][1]:.4f}",
            "fresh_last_used": f"{by[names[-2]][0] - by[names[-1]][0]:.4f}", "fresh_last_fresh": f"{by[names[-2]][1] - by[names[-1]][1]:.4f}"}, table


def branch_values():
    logs = [json.loads(p.read_text()) for p in sorted((ROOT / "study/fairness/branching").glob("*.json"))]
    if len(logs) < 2:
        return {"branch_note": "?"}
    ticks = {tuple(s["train_ticks"] for s in log["seeds"]) for log in logs}
    biggest = max(abs(a["bpb"] - b["bpb"]) for log in logs[1:] for a, b in zip(logs[0]["seeds"], log["seeds"]))
    if len(ticks) == 1:
        return {"branch_note": f"A probe entry whose ticks follow its floats, through a branch on `item()`, also read the same ticks on all {len(logs)} machines, with scores within {biggest:.1e}."}
    return {"branch_note": f"A probe entry whose ticks follow its floats, through a branch on `item()`, did not read the same ticks on all {len(logs)} machines, and its scores differed by up to {biggest:.1e}. So the claim holds for programs that do not branch on values, which covers all {len(list((ROOT / 'records').glob('[0-9]*')))} records."}


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
    return re.sub(r"\{(\w+)\}", lambda m: fill(values[m.group(1)], values), text)


def claims_rows(table):
    return [(t["attempt"].removeprefix("attempt-"), t["base"], f"{t['claimed_gain']:.4f}", f"{t['gain']:.4f} ± {t['ci95']:.4f}", t["outcome"], "yes" if t["record"] else "no") for t in table]


CLAIMS_HEAD = ("Attempt", "Started from", "Claimed gain", "Verified gain ± 95% half-width", "Outcome", "Record")
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
    return md_table(("", "Record", "Test bits per byte, mean ± seed SD on 8 seeds", "From"), body)


def put(text, key, body):
    return re.sub(rf"<!-- {key} -->.*?<!-- /{key} -->", lambda m: f"<!-- {key} -->\n{body}\n<!-- /{key} -->", text, flags=re.S)


def tex_table(head, rows, spec):
    esc = lambda c: str(c).replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$")
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(map(esc, head)) + r" \\", r"\midrule"]
    return "\n".join(lines + [" & ".join(map(esc, r)) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def tex_value(v):
    v = v.replace("_", "\\_").replace("%", "\\%")
    return re.sub(r"`([^`]*)`", r"\\texttt{\1}", v)


def paper(values, claims, fair_table, fresh_table, fresh_head):
    """Everything numeric in the paper comes through paper/numbers.tex and the two generated tables."""
    out = ROOT / "paper"
    macro = lambda key: "v" + key.replace("_", "").replace("003", "OOO").replace("004", "OOF")
    (out / "numbers.tex").write_text("".join(rf"\newcommand{{\{macro(k)}}}{{{tex_value(v)}}}" + "\n" for k, v in values.items()))
    (out / "claims.tex").write_text(tex_table(CLAIMS_HEAD, claims_rows(claims), "lllllc"))
    (out / "fair.tex").write_text(tex_table(FAIR_HEAD, fair_rows(fair_table), "lcccc"))
    (out / "fresh.tex").write_text(tex_table(fresh_head, fresh_table, "lccc"))


def main():
    rows = records()
    head = rows[0]
    values, claims = series_values()
    fair, fair_table = fair_values()
    values |= fair | quick_values()
    values |= {"attacks": str(sum(1 for p in (ROOT / "tests/attacks").glob("*.py") if p.stem != "legit_tiny")), "record_test": f"{head['test']:.4f}", "record_name": head["title"],
               "agent_records": str(sum(r["source"].startswith("agent") for r in rows)),
               "recipe_test": f"{min(r['test'] for r in rows if not r['source'].startswith('agent')):.4f}"}
    fresh, fresh_table = fresh_values(rows)
    values |= twin_values() | seed_time_values() | power_values() | fresh | branch_values() | {"record_count": str(len(rows))}
    FRESH_HEAD = ("Record", "Usual 1024 windows", "Next 1024 windows", "Difference")
    tables = {"claims_table": html_table(CLAIMS_HEAD, claims_rows(claims)), "fair_table": html_table(FAIR_HEAD, fair_rows(fair_table)), "fresh_table": html_table(FRESH_HEAD, fresh_table)}
    copy = json.loads((DOCS / "copy.json").read_text())
    data = {"records": rows, "lead": fill(copy["lead"], values), "html": {k: fill(v, values | tables) for k, v in copy["html"].items()}}
    (DOCS / "data.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    for name in ("__init__.py", "_engine.py", "_data.py"):
        shutil.copy2(ROOT / "odometer" / name, DOCS / "demo" / "odometer" / name)
    shutil.copy2(ROOT / head["code"], DOCS / "demo" / "train.py")

    tables = {"claims_table": md_table(CLAIMS_HEAD, claims_rows(claims)), "fair_table": md_table(FAIR_HEAD, fair_rows(fair_table)), "fresh_table": md_table(FRESH_HEAD, fresh_table)}
    readme = (ROOT / "README.md").read_text()
    readme = re.sub(r"\{(two_lo|two_hi|four_lo|four_hi|rule_003|rule_004|rep_power|twin_max|twin_runs)\}", lambda m: values[m.group(1)], readme)
    readme = put(readme, "records", records_md(rows))
    for key, body in copy["readme"].items():
        readme = put(readme, key, fill(body, values | tables))
    (ROOT / "README.md").write_text(readme)
    (ROOT / "site/writeup.md").write_text(fill((ROOT / "site/writeup.in.md").read_text(), values | tables))
    paper(values, claims, fair_table, fresh_table, FRESH_HEAD)
    print(f"{len(rows)} records, current {head['name']} {head['test']:.4f}")


if __name__ == "__main__":
    main()
