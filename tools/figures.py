"""Draw the figures from the logs, as SVG that follows the reader's colour scheme. No plotting library.

    python tools/figures.py
"""

import json
import math
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "fig"
STYLE = """<style>
.t{font:13px ui-sans-serif,-apple-system,'Segoe UI',Roboto,sans-serif;fill:#141412;font-variant-numeric:tabular-nums}
.m{fill:#62605a}.b{font-weight:600}.a{fill:#b92c17}
.l{fill:none;stroke:#141412;stroke-width:1.6}.la{stroke:#b92c17;stroke-width:2.4}.g{stroke:#d9d7cf;stroke-width:1;fill:none}.ax{stroke:#62605a;stroke-width:1;fill:none}
.p{fill:#141412}.pa{fill:#b92c17}.h{fill:#f6f5f1;stroke:#141412;stroke-width:1.6}.w{stroke:#141412;stroke-width:1.4;fill:none}.wa{stroke:#b92c17;stroke-width:1.8;fill:none}
@media (prefers-color-scheme:dark){.t{fill:#ecebe4}.m{fill:#9b998f}.a{fill:#ff8a6b}.l{stroke:#ecebe4}.la{stroke:#ff8a6b}.g{stroke:#2e2d2a}.ax{stroke:#9b998f}.p{fill:#ecebe4}.pa{fill:#ff8a6b}.h{fill:#111110;stroke:#ecebe4}.w{stroke:#ecebe4}.wa{stroke:#ff8a6b}}
</style>"""


def svg(name, w, h, body, title):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{title}"><title>{title}</title>{STYLE}{body}</svg>\n')


def text(x, y, s, cls="", anchor="start"):
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    return f'<text class="t {cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{s}</text>'


def marker(kind, x, y, cls):
    if kind == "diamond":
        return f'<path class="{cls}" d="M{x:.1f} {y - 6:.1f}L{x + 6:.1f} {y:.1f}L{x:.1f} {y + 6:.1f}L{x - 6:.1f} {y:.1f}Z"/>'
    if kind == "square":
        return f'<rect class="{cls}" x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8"/>'
    if kind == "triangle":
        return f'<path class="{cls}" d="M{x:.1f} {y - 5.5:.1f}L{x + 5.5:.1f} {y + 4:.1f}L{x - 5.5:.1f} {y + 4:.1f}Z"/>'
    return f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="4.2"/>'


def spread(ys, gap):
    """Push label heights apart by at least gap, keeping their order."""
    out = sorted(ys)
    for _ in range(50):
        for i in range(len(out) - 1):
            if out[i + 1] - out[i] < gap:
                mid = (out[i] + out[i + 1]) / 2
                out[i], out[i + 1] = mid - gap / 2, mid + gap / 2
    return out


SWEEP_NAMES = {"01-gpt": ("Small GPT", "circle", "p"), "02-rnn": ("Character RNN", "square", "h"), "03-modern": ("Modern transformer", "triangle", "p"),
               "04-ppm": ("PPM", "square", "p"), "05-mixer": ("Hashed mixing", "circle", "h"), "06-mixer-fold": ("Record 06, an agent's", "diamond", "pa")}


def sweep():
    runs = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "study" / "sweep").glob("*.json"))}
    W, H, L, R, T, B = 960, 500, 56, 190, 16, 44
    budgets = sorted({row["budget"] for rows in runs.values() for row in rows})
    X = lambda b: L + (math.log(b) - math.log(budgets[0])) / (math.log(budgets[-1]) - math.log(budgets[0])) * (W - L - R)
    lo, hi = 1.05, 4.0
    Y = lambda v: T + (math.log(hi) - math.log(v)) / (math.log(hi) - math.log(lo)) * (H - T - B)
    body = ""
    for v in (1.2, 1.5, 2, 3, 4):
        body += f'<path class="g" d="M{L} {Y(v):.1f}H{W - R}"/>' + text(L - 10, Y(v) + 4, f"{v:g}", "m", "end")
    for b in budgets:
        body += text(X(b), H - B + 22, f"{b / 1e9:g} billion", "m", "middle")
    body += f'<path class="ax" stroke-dasharray="3 4" d="M{X(4e10):.1f} {T}V{H - B}"/>' + text(X(4e10) + 8, T + 12, "the track's budget", "m")
    body += text(L, H - 6, "Training budget, in ticks", "m") + text(L, T + 12, "Dev bits per byte", "m")
    labels = []
    for name, rows in runs.items():
        label, kind, fill = SWEEP_NAMES.get(name, (name, "circle", "p"))
        means = [(r["budget"], statistics.mean(s["bpb"] for s in r["seeds"])) for r in rows]
        accent = fill == "pa"
        body += '<path class="l%s" d="%s"/>' % (" la" if accent else "", "".join(f"{'M' if i == 0 else 'L'}{X(b):.1f} {Y(v):.1f}" for i, (b, v) in enumerate(means)))
        for b, v in means:
            body += marker(kind, X(b), Y(v), fill)
        labels.append((Y(means[-1][1]), f"{label} {means[-1][1]:.2f}", "a b" if accent else ""))
    labels.sort()
    for y, (_, s, cls) in zip(spread([l[0] for l in labels], 17), labels):
        body += text(W - R + 12, y + 4, s, cls)
    svg("sweep.svg", W, H, body, "Dev bits per byte against training budget for each record")


def ablation():
    base = json.loads((ROOT / "records/05-mixer/log.json").read_text())["dev"]
    rec = json.loads((ROOT / "records/06-mixer-fold/log.json").read_text())["dev"]
    d = ROOT / "study/attempts/pilot-02/ablations"
    rows = [("The record it started from", base["mean_bpb"], base["ci95"], False), ("Lazy tables only", *[json.loads((d / "lazy_only.json").read_text())["summary"][k] for k in ("mean_bpb", "ci95")], False),
            ("Folded head only", *[json.loads((d / "fold_only.json").read_text())["summary"][k] for k in ("mean_bpb", "ci95")], False), ("Both, the agent's entry", rec["mean_bpb"], rec["ci95"], True)]
    W, H, L, R, T, B = 960, 220, 216, 96, 10, 40
    lo, hi = 1.142, 1.16
    X = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    body = ""
    for v in (1.145, 1.15, 1.155, 1.16):
        body += f'<path class="g" d="M{X(v):.1f} {T}V{H - B}"/>' + text(X(v), H - B + 20, f"{v:.3f}", "m", "middle")
    body += text(L, H - 2, "Dev bits per byte, 8 seeds each, with 95% intervals. Lower is better.", "m")
    for i, (label, mean, ci, accent) in enumerate(rows):
        y = T + 22 + i * 42
        body += text(L - 14, y + 4, label, "a b" if accent else "", "end")
        body += f'<path class="{"wa" if accent else "w"}" d="M{X(mean - ci):.1f} {y}H{X(mean + ci):.1f}"/>' + marker("diamond" if accent else "circle", X(mean), y, "pa" if accent else "p")
        body += text(W - R + 12, y + 4, f"{mean:.4f}", "a b" if accent else "")
    svg("ablation.svg", W, H, body, "How the agent's gain splits between its two changes")


def ladder():
    rows = []
    names = {"01-gpt": "Small GPT", "02-rnn": "Character RNN", "03-modern": "Modern transformer", "04-ppm": "PPM", "05-mixer": "Hashed mixing", "06-mixer-fold": "Folded head, lazy tables"}
    for d in sorted((ROOT / "records").glob("[0-9]*")):
        log = json.loads((d / "log.json").read_text())
        rows.append((names.get(d.name, d.name), log["confirmation_summary"]["mean_bpb"], log["source"].startswith("agent")))
    W, H, L, R, T, B = 960, 320, 56, 150, 18, 34
    lo, hi = 1.05, 1.8
    X = lambda i: L + (i + 0.5) / len(rows) * (W - L - R)
    Y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    body = ""
    for v in (1.2, 1.4, 1.6, 1.8):
        body += f'<path class="g" d="M{L} {Y(v):.1f}H{W - R}"/>' + text(L - 10, Y(v) + 4, f"{v:.1f}", "m", "end")
    body += text(L, H - 6, "Test bits per byte on 8 fresh seeds. Seed spread is smaller than the marks.", "m")
    path = "".join((f"M{X(i):.1f} {Y(v):.1f}" if i == 0 else f"V{Y(v):.1f}") + f"H{X(i + 1) if i < len(rows) - 1 else W - R:.1f}" for i, (_, v, _a) in enumerate(rows))
    body += f'<path class="l" d="{path}"/>'
    for i, (name, v, agent) in enumerate(rows):
        body += marker("diamond" if agent else "circle", X(i), Y(v), "pa" if agent else "p")
        body += text(X(i) + 8, Y(v) - 9, f"{name}", "a b" if agent else "b") + text(X(i) + 8, Y(v) + 18, f"{v:.4f}" + ("  agent" if agent else "  recipe"), "a" if agent else "m")
    svg("ladder.svg", W, H, body, "The record table: test bits per byte falling with each new record")


def claims():
    path = ROOT / "study" / "results.json"
    if not path.exists():
        return
    table = json.loads(path.read_text())
    rows = [dict(t, pilot=False) for t in table["claims_table"]] + [dict(t, pilot=True) for t in table.get("pilots", [])]
    rows = [t for t in rows if t.get("gain") is not None and t.get("claimed_gain") is not None]
    if not rows:
        return
    W, H, L, R, T, B = 960, 60 + 30 * len(rows), 150, 96, 12, 44
    lo = min(0.0, *[t["gain"] - t["ci95"] for t in rows], *[t["claimed_gain"] for t in rows])
    hi = max(*[t["gain"] + t["ci95"] for t in rows], *[t["claimed_gain"] for t in rows]) * 1.04
    X = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    body = f'<path class="g" d="M{X(0):.1f} {T}V{H - B}"/>' + text(X(0), H - B + 18, "0", "m", "middle")
    body += text(L, H - 6, "Gain over the starting record, bits per byte. Filled mark: verified on 8 fresh seeds, with its 95% interval. Open mark: what the agent claimed.", "m")
    for i, t in enumerate(rows):
        y = T + 16 + i * 30
        body += text(L - 14, y + 4, f"{t['attempt']} claim {t['n']}", "m", "end")
        body += f'<path class="w" d="M{X(t["gain"] - t["ci95"]):.1f} {y}H{X(t["gain"] + t["ci95"]):.1f}"/>' + marker("circle", X(t["claimed_gain"]), y, "h") + marker("circle", X(t["gain"]), y, "p")
        body += text(W - R + 12, y + 4, t["outcome"], "b" if t["outcome"] == "replicated" else "m")
    svg("claims.svg", W, H, body, "What each agent claim claimed against what the rerun found")


if __name__ == "__main__":
    sweep()
    ablation()
    ladder()
    claims()
    print(sorted(p.name for p in OUT.glob("*.svg")))
