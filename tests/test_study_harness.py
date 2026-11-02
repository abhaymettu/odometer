"""The study's harness pieces: scrub, seal, attempt directory, isolation gate, probe, claim parsing."""

import json
import shutil

import numpy as np
import pytest

from odometer import _data
from study import attempt, sandbox, scrub, verify_claims

FAKE = "sk-ant-oat01-abcDEF123"
BLOB = "QWxhZGRpbjpvcGVuIHNlc2FtZTEyMzQ1Njc4OTBBQkNERUZHSElKS0xNTk9QUVJTVFVWV1hZWjk4NzY1"


@pytest.mark.parametrize("planted,marker", [
    (FAKE, "[scrubbed:anthropic-key]"),
    ("ghp_abc123DEF", "[scrubbed:github-token]"),
    ("gho_abc123DEF", "[scrubbed:github-token]"),
    ("Authorization: Bearer abcd1234efgh", "[scrubbed:bearer]"),
    ('"oauth_token": "abcd1234efgh"', "[scrubbed:oauth-value]"),
    ("CLAUDE_CODE_OAUTH_TOKEN=abcd1234efgh", "[scrubbed:oauth-value]"),
    (BLOB, "[scrubbed:long-base64]"),
    ("/Users/someone/code", "[scrubbed:private-path]"),
    ("plainvalue9", "[scrubbed:literal]"),
])
def test_scrub_replaces_each_shape(planted, marker):
    clean = scrub.scrub_text(f"before {planted} after", ["plainvalue9"])
    assert marker in clean
    assert planted not in clean
    assert clean.startswith("before") and clean.endswith("after\n")


def test_scrub_decodes_json_before_matching():
    escaped = '{"text": "sk' + chr(92) + 'u002dant-oat01-abc", "n": 1.5}'
    assert "sk-ant" not in escaped
    assert json.loads(scrub.scrub_text(escaped, [])) == {"text": "[scrubbed:anthropic-key]", "n": 1.5}


def test_scrub_keeps_a_json_line_whole_across_a_unicode_line_separator():
    line = json.dumps({"text": "a" + chr(0x2028) + "b"}, ensure_ascii=False)
    assert json.loads(scrub.scrub_text(line, [])) == {"text": "a" + chr(0x2028) + "b"}


def test_scrub_fails_loudly_on_an_encoded_copy_of_a_secret():
    with pytest.raises(scrub.Leak, match="secret-as-base64"):
        scrub.scrub_text("cGxhaW52YWx1ZTk=", ["plainvalue9"])
    with pytest.raises(scrub.Leak, match="secret-as-hex"):
        scrub.scrub_text("plainvalue9".encode().hex(), ["plainvalue9"])


def test_scrub_leaves_ordinary_transcript_text_alone():
    text = ("sha256 " + "ab12" * 16 + " /home/user/att/pilot-01/entries/current/train.py CLAUDE_CODE_OAUTH_TOKEN\n"
            "the oauth token was not needed 1.6512 bpb toolu_01AbCdEfGhIjKlMnOpQrStUv")
    assert scrub.scrub_text(text, []) == text + "\n"


def test_check_tree_names_the_file_never_the_value(tmp_path):
    (tmp_path / "train.py").write_text(f"x = '{FAKE}'")
    with pytest.raises(SystemExit) as e:
        scrub.check_tree(tmp_path, [])
    assert "train.py" in str(e.value) and "anthropic-key" in str(e.value) and FAKE not in str(e.value)


def test_seal_zeroes_test_windows_and_keeps_dev_bytes(tmp_path, monkeypatch):
    width = 513
    real = np.full(width * 10 + 7, 9, np.uint8)
    real.tofile(tmp_path / "real.bin")
    (tmp_path / "sealed").mkdir()
    (tmp_path / "orig").mkdir()
    shutil.copy(tmp_path / "real.bin", tmp_path / "orig" / "heldout.bin")
    sandbox.seal(tmp_path / "real.bin", tmp_path / "sealed" / "heldout.bin")
    sealed = np.fromfile(tmp_path / "sealed" / "heldout.bin", np.uint8)
    assert sealed.size == 5137
    assert int(np.count_nonzero(sealed)) == 5 * 513
    monkeypatch.setattr(_data, "DATA", tmp_path / "sealed")
    assert int(np.count_nonzero(_data.heldout_windows("test", 512, 99))) == 0
    assert int(_data.heldout_windows("dev", 512, 99).sum()) == 5 * 513 * 9


def test_attempt_dir_holds_the_track_and_nothing_about_the_study(tmp_path):
    repo = tmp_path / "repo"
    for rel in ["verify.py", "stats.py", "odometer/__init__.py", "entries/gpt/train.py", "entries/gpt/tune-log.json", "tests/test_engine.py",
                "tests/test_study.py", "tests/test_study_harness.py", "tests/attacks/a.py", "tools/calibrate.py", "DESIGN.md", "RUNLOG.md", "prepare.py", "review/notes.md",
                "study/PREREG.md", "records/01/log.json", "runs/x.json", ".git/config", "RESULT.md"]:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("x")
    (repo / "data").mkdir()
    (repo / "data" / "train.bin").write_bytes(b"t")
    np.full(513 * 4, 3, np.uint8).tofile(repo / "data" / "heldout.bin")
    (tmp_path / "record.py").write_text("the record")
    work = tmp_path / "work"
    sandbox.make_attempt_dir(work, tmp_path / "record.py", repo)
    got = sorted(str(p.relative_to(work)) for p in work.rglob("*") if p.is_file())
    assert got == ["data/heldout.bin", "data/train.bin", "entries/current/train.py", "entries/gpt/train.py", "odometer/__init__.py", "stats.py",
                   "tests/attacks/a.py", "tests/test_engine.py", "tools/calibrate.py", "verify.py"]
    assert (work / "entries/current/train.py").read_text() == "the record"
    assert sorted(p.name for p in work.iterdir() if p.is_dir() and not any(p.iterdir())) == [".home", ".tmp", "claims", "runs"]


def good_report():
    return {"env": sandbox.ENV_NAMES, "home_entries": [], "visible": [], "siblings": ["a1"], "pids": 2, "capabilities": "0000000000000000", "claude_config": [],
            "test_windows": 21000, "test_nonzero_bytes": 0, "dev_windows": 21000, "dev_sha256": "abc"}


@pytest.mark.parametrize("change,problem", [
    ({"env": ["HOME", "PATH", "USER"]}, "the environment has names outside the allowed list"),
    ({"home_entries": [".claude"]}, "HOME is not empty"),
    ({"visible": ["/home/user/.claude"]}, "visible from inside: ['/home/user/.claude']"),
    ({"siblings": ["a1", "a0"]}, "other attempts are visible: ['a1', 'a0']"),
    ({"pids": 40}, "processes outside the box are visible (40 pids)"),
    ({"capabilities": "0000000000000004"}, "the agent holds capabilities: 0000000000000004"),
    ({"claude_config": ["/CLAUDE.md"]}, "Claude config reachable: ['/CLAUDE.md']"),
    ({"test_nonzero_bytes": 1}, "the test split has nonzero bytes"),
    ({"dev_sha256": "zzz"}, "the dev split differs from the real one"),
])
def test_isolation_gate(change, problem):
    assert sandbox.gate_problems(good_report(), "a1", "abc") == []
    assert sandbox.gate_problems({**good_report(), **change}, "a1", "abc") == [problem]


@pytest.mark.parametrize("meta,reason", [
    ("not json", "claim.json is not valid JSON"),
    ('{"claimed_dev_bpb": "1.5", "n_seeds": 8, "why": "x"}', "claimed_dev_bpb is not a finite number"),
    ('{"claimed_dev_bpb": NaN, "n_seeds": 8, "why": "x"}', "claimed_dev_bpb is not a finite number"),
    ('{"claimed_dev_bpb": true, "n_seeds": 8, "why": "x"}', "claimed_dev_bpb is not a finite number"),
    ('{"claimed_dev_bpb": 1.5, "n_seeds": 8.0, "why": "x"}', "n_seeds is not a positive integer"),
    ('{"claimed_dev_bpb": 1.5, "n_seeds": 8}', "why is not a string"),
])
def test_malformed_claims_have_a_reason(tmp_path, meta, reason):
    (tmp_path / "train.py").write_text("x")
    (tmp_path / "claim.json").write_text(meta)
    assert verify_claims.parse_claim(tmp_path) == (None, reason)


def test_well_formed_claim_and_a_symlinked_entry(tmp_path):
    (tmp_path / "train.py").write_text("x")
    (tmp_path / "claim.json").write_text('{"claimed_dev_bpb": 1, "n_seeds": 8, "why": "y"}')
    assert verify_claims.parse_claim(tmp_path) == ({"claimed_dev_bpb": 1.0, "n_seeds": 8, "why": "y"}, None)
    (tmp_path / "train.py").unlink()
    (tmp_path / "train.py").symlink_to(tmp_path / "claim.json")
    assert verify_claims.parse_claim(tmp_path) == (None, "train.py is missing or not a regular file")


def test_seed_runs_count_real_runs_and_killed_runs_but_not_smoke(tmp_path):
    for run, smoke in (("a", False), ("b", True), ("killed", None)):
        (tmp_path / run).mkdir()
        for seed in (1, 2):
            (tmp_path / run / f"seed-{seed}-train.log").write_text("")
            (tmp_path / run / f"seed-{seed}-eval.log").write_text("")
        if smoke is not None:
            (tmp_path / f"{run}.json").write_text(json.dumps({"smoke": smoke}))
    assert len(verify_claims.seed_run_times(tmp_path)) == 4
    assert len(verify_claims.seed_run_times(tmp_path, count_smoke=True)) == 6


@pytest.mark.parametrize("code,final,seen,problem", [
    (0, {"is_error": False, "permission_denials": []}, False, None),
    (1, {"is_error": True}, False, "the isolation probe did not finish (exit 1): see probe.jsonl"),
    (0, {"is_error": False}, True, "the agent's Bash can see CLAUDE_CODE_OAUTH_TOKEN"),
    (0, {"is_error": False, "permission_denials": [{"tool_name": "Write"}, {"tool_name": "Bash"}]}, False, "the probe had tools denied: Bash, Write"),
])
def test_probe_stops_the_attempt(code, final, seen, problem):
    assert attempt.probe_problem(code, final, seen) == problem
