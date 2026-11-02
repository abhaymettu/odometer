"""One whole series through the real pipeline with a stand-in agent, on the verifier's smoke budget.

    ODOMETER_E2E=1 pytest tests/test_study_e2e.py -s    # needs bwrap; about 3 minutes on 2 cores
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

import verify
from study import sandbox
from study.common import ATT_ROOT, ROOT, layout, read_json


def run(*cmd, **env):
    proc = subprocess.run([sys.executable, "-m", *cmd], cwd=ROOT, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(f"$ {' '.join(cmd)}\n{proc.stdout}")
    return proc


@pytest.mark.skipif(not os.environ.get("ODOMETER_E2E") or not shutil.which("bwrap"), reason="set ODOMETER_E2E=1 on a machine with bwrap")
def test_end_to_end_with_a_stand_in_agent(tmp_path):
    token = "plainfaketokenvalue42"
    place = layout(smoke=True)
    shutil.rmtree(place.base, ignore_errors=True)
    for name in ("e2e", "e2e-hang", "e2e-die", "e2e-probe", "e2e-denied"):
        shutil.rmtree(ATT_ROOT / name, ignore_errors=True)
    smoke = ("--smoke", "--jobs", "2")
    (tmp_path / "review.md").write_text("Reviewed by hand.")
    assert run("study.ladder", "01-gpt", "entries/gpt/train.py", "--review", str(tmp_path / "review.md"), *smoke).returncode == 0

    agent = ("--smoke", "--path-prefix", "study/standin")
    assert run("study.attempt", "e2e", "--model", "standin", *agent, CLAUDE_CODE_OAUTH_TOKEN=token).returncode == 0
    out = place.attempts / "e2e"
    report = read_json(out / "isolation.json")
    assert report["env"] == sandbox.ENV_NAMES and report["home_entries"] == [] and report["visible"] == [] and report["claude_config"] == []
    assert report["test_nonzero_bytes"] == 0 and report["test_windows"] > 20000 and report["pids"] <= 3 and int(report["capabilities"], 16) == 0
    assert report["dev_windows"] > 20000 and report["siblings"] == ["e2e"]
    sealed = np.fromfile(ATT_ROOT / "e2e" / "data" / "heldout.bin", np.uint8)
    assert sealed.size == (ROOT / "data" / "heldout.bin").stat().st_size and 0 < np.count_nonzero(sealed) < sealed.size

    transcript = (out / "transcript.jsonl").read_text()
    for planted in (token, "sk-ant-oat01", "ghp_", "fakeBearer", "QWxhZGRpbjpv"):
        assert planted not in transcript and planted not in (out / "probe.jsonl").read_text()
    for marker in ("anthropic-key", "github-token", "bearer", "long-base64", "literal"):
        assert f"[scrubbed:{marker}]" in transcript
    meta = read_json(out / "attempt.json")
    assert meta["status"] == "finished" and meta["final_result"]["total_cost_usd"] == 0.0 and meta["claude_version"] == "0.0.0 (stand-in)"
    assert meta["init"]["model"] == "standin" and "result" not in meta["final_result"]

    assert run("study.verify_claims", "e2e", *smoke, CLAUDE_CODE_OAUTH_TOKEN=token).returncode == 0
    result = read_json(out / "result.json")
    by_n = {c["n"]: c for c in result["claims"]}
    assert result["ignored"] == ["4"]
    assert by_n[1]["outcome"] != "rejected" and by_n[1]["seed_runs_before"] == 4 and by_n[1]["code_sha256"] == verify.code_hash()
    assert by_n[1]["verdict"]["stage1"]["gain"] == pytest.approx(by_n[1]["gain"])
    assert by_n[2]["outcome"] == "rejected" and "import os" in by_n[2]["reason"]
    assert by_n[3] == {**by_n[3], "outcome": "rejected", "reason": "claim.json is not valid JSON"}
    before = (out / "claims" / "1" / "outcome.json").read_text()
    assert run("study.verify_claims", "e2e", *smoke, CLAUDE_CODE_OAUTH_TOKEN=token).returncode == 0
    assert (out / "claims" / "1" / "outcome.json").read_text() == before

    assert run("study.attempt", "e2e-hang", "--model", "standin-hang", "--minutes", "0.1", *agent, CLAUDE_CODE_OAUTH_TOKEN=token).returncode == 0
    assert read_json(place.attempts / "e2e-hang" / "attempt.json")["exit_code"] == 124
    assert subprocess.run(["pgrep", "-f", "sleep 3600"], capture_output=True).stdout == b""
    assert run("study.attempt", "e2e-die", "--model", "standin-die", *agent, CLAUDE_CODE_OAUTH_TOKEN=token).returncode == 0
    assert read_json(place.attempts / "e2e-die" / "attempt.json")["status"] == "infra"
    assert run("study.attempt", "e2e-denied", "--model", "standin-denied", "--probe-only", *agent, CLAUDE_CODE_OAUTH_TOKEN=token).returncode == 0
    assert read_json(place.attempts / "e2e-denied" / "attempt.json")["reason"] == "the probe had tools denied: Bash"
    shutil.rmtree(place.attempts / "e2e-denied")
    assert run("study.verify_claims", "e2e-hang", *smoke, CLAUDE_CODE_OAUTH_TOKEN=token).returncode == 0
    assert run("study.verify_claims", "e2e-die", *smoke, CLAUDE_CODE_OAUTH_TOKEN=token).returncode == 0

    assert run("study.null_control", "--nulls", "2", *smoke, CLAUDE_CODE_OAUTH_TOKEN=token).returncode == 0

    entry = str(out / "claims" / "1" / "train.py")
    refused = run("study.ladder", "02-claim", entry, "--review", str(tmp_path / "review.md"), *smoke)
    assert refused.returncode != 0 and "--verdict is required: 01-gpt is the record to beat" in refused.stdout
    verdict = {"sha256": by_n[1]["sha256"], "verdict": {"record": True, "against": str(out / "base.json")}}
    (tmp_path / "verdict.json").write_text(json.dumps(verdict))
    assert run("study.ladder", "02-claim", entry, "--review", str(tmp_path / "review.md"), "--verdict", str(tmp_path / "verdict.json"), *smoke).returncode == 0
    assert read_json(place.current)["replaces"] == "01-gpt"
    assert run("study.attempt", "e2e-probe", "--model", "standin", "--probe-only", *agent, CLAUDE_CODE_OAUTH_TOKEN=token).returncode == 0
    assert read_json(place.attempts / "e2e-probe" / "base.json")["name"] == "02-claim"
    text = run("study.analyze", "--smoke").stdout
    results = read_json(place.base / "results.json")
    assert "SMOKE RUN" in text
    assert results["attempts"] == {"started": 3, "counted": 2, "excluded": ["e2e-die"], "pending": [], "with_no_claims": 1, "with_tool_denials": 0}
    assert results["claims"]["total"] == 3 and results["claims"]["rejected"] == 2
    assert results["null_control"]["01-gpt"]["nulls"] == 2
