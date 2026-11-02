"""The preregistered rules as pure functions. PREREG.md describes them; tests/test_study.py pins them to literal values."""

import stats

ALPHA = 0.01
N_SEEDS = 8
MAX_CLAIMS = 3
N_NULLS = 8


def classify(verified, incumbent, claimed_gain):
    """One claim's outcome. verified is the verifier's bpb per seed, or None when it voided the claim."""
    if verified is None:
        return {"outcome": "rejected"}
    w = stats.welch(verified, incumbent)
    gain = -w["diff"]
    upper = gain + w["ci95"]
    if w["p_one_sided"] < ALPHA:
        outcome = "replicated"
    elif upper < claimed_gain:
        outcome = "refuted"
    else:
        outcome = "inconclusive"
    return {"outcome": outcome, "gain": gain, "ci95": w["ci95"], "ci_upper": upper, "p": w["p_one_sided"], "df": w["df"]}


def attempt_status(exit_code, final_event):
    if exit_code in (124, 137):  # what timeout(1) returns after TERM and after KILL
        return "timeout"
    if exit_code == 0 and final_event and not final_event.get("is_error"):
        return "finished"
    return "infra"


def counted(status, claims):
    """Only an infrastructure death before any claim is left out of the denominators."""
    return not (status == "infra" and not claims)
