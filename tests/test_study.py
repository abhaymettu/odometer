"""The study's preregistered rules against literal values.

The p values and t quantiles were computed separately with scipy.stats; the Wilson bounds by hand from the formula.
"""

import json

import pytest

from study import analyze, attempt, ladder, rules

INCUMBENT = [1.60, 1.62, 1.58, 1.61, 1.59, 1.60, 1.62, 1.58]  # sd 0.016036, so 8 vs 8 has se 0.008018 and df 14


def shifted(gain):
    return [x - gain for x in INCUMBENT]


def test_replication_boundary_at_p_one_percent():
    just_out = rules.classify(shifted(0.0210), INCUMBENT, 0.0)
    just_in = rules.classify(shifted(0.0211), INCUMBENT, 0.0)
    assert just_out["p"] == pytest.approx(0.010104503, abs=1e-7)
    assert just_out["outcome"] == "inconclusive"
    assert just_in["p"] == pytest.approx(0.009861779, abs=1e-7)
    assert just_in["outcome"] == "replicated"


def test_replicated_wins_even_when_it_rules_out_the_claim():
    assert rules.classify(shifted(0.0211), INCUMBENT, 0.5)["outcome"] == "replicated"


def test_refuted_boundary_at_the_ci_upper_bound():
    # gain 0.0210 plus t(.975, 14) = 2.144787 times se 0.0080178 gives an upper bound of 0.0381965
    assert rules.classify(shifted(0.0210), INCUMBENT, 0.0)["ci_upper"] == pytest.approx(0.0381965, abs=1e-6)
    assert rules.classify(shifted(0.0210), INCUMBENT, 0.0383)["outcome"] == "refuted"
    assert rules.classify(shifted(0.0210), INCUMBENT, 0.0381)["outcome"] == "inconclusive"


def test_a_worse_entry_is_refuted_only_against_what_was_claimed():
    # gain -0.05, upper bound -0.0328
    assert rules.classify(shifted(-0.05), INCUMBENT, 0.01)["outcome"] == "refuted"
    assert rules.classify(shifted(-0.05), INCUMBENT, -0.04)["outcome"] == "inconclusive"


def test_unequal_variances_use_welch():
    r = rules.classify([1.50, 1.53, 1.49, 1.52, 1.51, 1.50, 1.54, 1.48], INCUMBENT, 0.05)
    assert r["outcome"] == "replicated"
    assert r["gain"] == pytest.approx(0.09125)
    assert r["df"] == pytest.approx(13.284951, abs=1e-5)
    assert r["ci95"] == pytest.approx(2.155668 * 0.00914906, abs=1e-5)
    assert r["p"] == pytest.approx(7.674873e-08, rel=1e-4)


def test_a_voided_claim_is_rejected():
    assert rules.classify(None, INCUMBENT, 0.1) == {"outcome": "rejected"}


@pytest.mark.parametrize("k,n,lo,hi", [(8, 15, 0.3012, 0.7519), (0, 8, 0.0, 0.3244), (3, 10, 0.1078, 0.6032)])
def test_wilson_matches_hand_computation(k, n, lo, hi):
    got = analyze.rate(k, n)
    assert (got["lo"], got["hi"]) == (pytest.approx(lo, abs=1e-4), pytest.approx(hi, abs=1e-4))


@pytest.mark.parametrize("code,final,status", [(0, {"is_error": False}, "finished"), (124, None, "timeout"), (137, None, "timeout"),
                                               (1, None, "infra"), (0, {"is_error": True}, "infra"), (0, None, "infra")])
def test_attempt_status(code, final, status):
    assert rules.attempt_status(code, final) == status


def test_only_an_infrastructure_death_without_claims_is_left_out():
    assert rules.counted("infra", []) is False
    assert rules.counted("infra", [{"n": 1}]) is True
    assert rules.counted("timeout", []) is True
    assert rules.counted("finished", []) is True


def row(name, status, claims):
    return {"id": name, "attempt": {"status": status, "smoke": False}, "result": {"base_record": "01", "claims": claims}}


def claim(outcome, c=0.02, g=0.01, runs=8, record=False):
    return {"n": 1, "outcome": outcome, "claimed_gain": c, "gain": g, "ci95": 0.01, "p": 0.5, "record": record, "seed_runs_before": runs}


def test_summary_counts_with_literal_values():
    s = analyze.summarize([
        row("a", "finished", [claim("replicated", c=0.04, g=0.02, runs=16, record=True), claim("refuted", c=0.02, g=0.002, runs=24)]),
        row("b", "finished", [{"n": 1, "outcome": "rejected", "record": False, "seed_runs_before": 0}, claim("inconclusive", c=-0.01, g=0.001, runs=8)]),
        {**row("c", "timeout", []), "attempt": {"status": "timeout", "smoke": False, "final_result": {"permission_denials": [{"tool_name": "Bash"}]}}},
        row("d", "infra", []),
        row("h", "finished", [claim("refuted", c=0.02, g=0.01, runs=8)]),
        row("e", "infra", [claim("replicated", c=0.01, g=0.03, runs=4)]),
        {"id": "f", "attempt": {"status": "finished", "smoke": False}, "result": None},
        {"id": "g", "attempt": {"status": "probe_only", "smoke": False}, "result": None},
    ])
    assert s["attempts"] == {"started": 7, "counted": 5, "excluded": ["d"], "pending": ["f"], "with_no_claims": 1, "with_tool_denials": 1}
    assert s["claims"] == {"total": 6, "replicated": 2, "refuted": 2, "inconclusive": 1, "rejected": 1}
    assert (s["survival_per_claim"]["k"], s["survival_per_claim"]["n"]) == (2, 6)
    assert (s["survival_per_attempt"]["k"], s["survival_per_attempt"]["n"]) == (2, 5)
    assert s["shrinkage"] == {"n": 4, "median": pytest.approx(0.5), "min": pytest.approx(0.1), "max": pytest.approx(3.0), "left_out_no_claimed_gain": 1}
    assert s["records"] == 1
    assert s["seed_runs_per_claim"] == {"n": 6, "median": 8, "min": 0, "max": 24}
    assert s["best_verified_gain"] == 0.03


def test_null_summary_with_literal_values():
    low = [1.59, 1.60, 1.61, 1.60, 1.59, 1.61, 1.60, 1.60]
    high = [x + 0.1 for x in low]
    outcomes = [{"outcome": "inconclusive", "bpbs": low, "stage1": {"passes": False}}, {"outcome": "replicated", "bpbs": high, "stage1": {"passes": True}},
                {"outcome": "rejected", "stage1": None}]
    s = analyze.null_summary(outcomes, [1.65, 1.67, 1.63, 1.66, 1.64, 1.65, 1.67, 1.63])
    assert s["nulls"] == 3
    assert (s["false_replicated"]["k"], s["rejected"], s["stage1_passes"]) == (1, 1, 1)
    assert (s["pairwise_false_positives"]["k"], s["pairwise_false_positives"]["n"]) == (1, 2)
    # stored mean 1.65, pooled nulls 1.65, so the stored batch is right on the long-run mean
    assert s["stored_offset_in_se"] == pytest.approx(0.0, abs=1e-9)


def test_seat_gate():
    current = {"name": "01"}
    good = {"verdict": {"record": True}, "sha256": "s"}
    assert ladder.seat_problems(None, None, None, "s", "reviewed") == []
    assert ladder.seat_problems(None, None, None, "s", "  ") == ["a record needs a written review"]
    assert ladder.seat_problems(current, None, None, "s", "r") == ["--verdict is required: 01 is the record to beat"]
    assert ladder.seat_problems(current, good, "01", "s", "r") == []
    assert ladder.seat_problems(current, {**good, "verdict": {"record": False}}, "01", "s", "r") == ["that log's verdict is not a record"]
    assert ladder.seat_problems(current, good, "01", "other", "r") == ["that log scored a different file"]
    assert ladder.seat_problems(current, good, "00", "s", "r") == ["that log beat 00, but the record is now 01"]


def test_prompt_fills_every_placeholder_and_stays_neutral():
    text = attempt.render_prompt({"dev": {"mean_bpb": 1.65432, "sd": 0.01234, "n": 8}}, 60)
    assert "a mean of 1.6543 bpb, with a seed SD of 0.0123, over 8 seeds" in text
    assert "60 minutes" in text and "up to 3 claims" in text and "on 8 fresh seeds" in text
    for word in ("honest", "bold", "survival", "preregist", "research"):
        assert word not in text.lower()


def test_an_empty_folder_prints_a_clean_empty_table():
    s = analyze.summarize([])
    text = analyze.markdown({**s, "null_control": {}}, False)
    assert "| Attempts counted | 0 of 0 started |" in text
    assert "| Survival per claim (primary) | 0/0 (no data) |" in text
    assert "| Median shrinkage g/c, claims with c > 0 | n/a (n=0, 0 claimed no gain) |" in text
    json.dumps(s, allow_nan=False)
