"""How much could the series' numbers have come out otherwise? A simulation with the real tests and each claim's seed SDs.

    python tools/power.py

Normal seed noise, 8 seeds a side, using stats.welch, rules.classify's alpha and verify.judge's margin, stage 1 then stage 2. Writes study/power.json.
"""

import json
import pathlib
import random
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import stats  # noqa: E402
from study import rules  # noqa: E402
from verify import RECORD_ALPHA, RECORD_MARGIN  # noqa: E402

TRIALS = 4_000
SD = 0.001
GAINS = (0.001, 0.002, 0.003, 0.004, 0.005)


def batch(rng, mean, sd):
    return [rng.gauss(mean, sd) for _ in range(8)]


def replicates(rng, gain, sd_claim, sd_base):
    return stats.welch(batch(rng, -gain, sd_claim), batch(rng, 0.0, sd_base))["p_one_sided"] < rules.ALPHA


def record_stage(rng, gain, sd, base):
    return stats.welch(batch(rng, -gain, sd), [b - RECORD_MARGIN for b in base])["p_one_sided"] < RECORD_ALPHA


def record_power(rng, gain, sd):
    wins = 0
    for _ in range(TRIALS):
        base = batch(rng, 0.0, sd)
        wins += record_stage(rng, gain, sd, base) and record_stage(rng, gain, sd, base)
    return wins / TRIALS


def all_replicate(rng, claims, scale):
    return sum(all(replicates(rng, c["claimed"] * scale, c["sd"], c["sd_base"]) for c in claims) for _ in range(TRIALS)) / TRIALS


def main():
    rng = random.Random(20261102)
    claims = []
    for path in sorted((ROOT / "study/attempts").glob("attempt-*/claims/*/verify.json")):
        v = json.loads(path.read_text())
        out = json.loads((path.parent / "outcome.json").read_text())
        base = json.loads((ROOT / "records" / json.loads((path.parent.parent.parent / "attempt.json").read_text())["base_record"] / "log.json").read_text())
        claims.append({"claimed": out["claimed_gain"], "sd": statistics.stdev(s["bpb"] for s in v["seeds"]), "sd_base": base["confirmation_summary"]["sd"]})
    result = {"trials": TRIALS, "claims": len(claims), "all_replicate_if_exact": all_replicate(rng, claims, 1.0), "all_replicate_if_half": all_replicate(rng, claims, 0.5),
              "replicated_power_one_sd": sum(replicates(rng, SD, SD, SD) for _ in range(TRIALS)) / TRIALS,
              "record_rule_sd": SD, "record_rule_power": {f"{g:g}": record_power(rng, g, SD) for g in GAINS}}
    (ROOT / "study/power.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
