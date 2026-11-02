"""Every file in tests/attacks/ is an entry written to cheat. This suite proves the verifier refuses each one, for the
exact reason it should, and that an honest entry still passes. README.md beside the attacks lists what each one found.
"""

import json
import pathlib
import re
import subprocess
import sys

import pytest

import verify

ROOT = pathlib.Path(verify.__file__).resolve().parent
ATTACKS = ROOT / "tests" / "attacks"
SEED = 11
HONEST_TICKS = 205_374_880  # the shared 40-step bigram training, priced by hand in test_honest_training_price
TWIN_FAILED = r"void: twin check failed \(logits before the cut moved by \d\.\d\de[-+]\d\d nats\)"
LINE_CAP = "void: rule violation: more than 20000000 Python lines and jumps in the entry"
BAD_LOGITS = "void: rule violation: forward() must return float logits of shape (batch, 512, 256)"
OVER_CAP = "void: more than 4194304 parameters, with one more per character of their names"


def source(name):
    return (ATTACKS / f"{name}.py").read_text(encoding="utf-8")


def smoke(name, tmp_path):
    """The real command line, so both real child processes run: verify.py <attack> --smoke --seeds 11,"""
    out = tmp_path / f"{name}.json"
    cmd = [sys.executable, "verify.py", str(ATTACKS / f"{name}.py"), "--smoke", "--seeds", f"{SEED},", "--split", "dev",
           "--out", str(out)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return proc, json.loads(out.read_text())["seeds"][0], out.with_suffix("")


def children(name, tmp_path, seed=SEED, **track):
    """The same two child processes under a changed track, for what the command line cannot set."""
    return verify.run_seed(ATTACKS / f"{name}.py", seed, "dev", {**verify.TRACK, **verify.SMOKE, **track}, tmp_path)


def test_every_attack_is_exercised_here():
    text = pathlib.Path(__file__).read_text()
    assert [p.stem for p in ATTACKS.glob("*.py") if f'"{p.stem}"' not in text] == []


# ---------------------------------------------------------------- caught by the static rules

STATIC = {
    "peek_open": ["line 9: builtin open is not on the allowed list", "line 9: .read is not on the allowed list"],
    "peek_numpy": ["line 9: import numpy", "line 10: .fromfile is not on the allowed list",
                   "line 10: .uint8 is not on the allowed list"],
    "peek_odometer_data": ["line 7: from odometer import _data", "line 11: ._data is not on the allowed list",
                           "line 11: .heldout_windows is not on the allowed list"],
    "peek_match_pattern": ["line 9: Match is not allowed", "line 10: .Tensor is not on the allowed list"],
    "peek_generator_frame": [
        "line 10: .gi_frame is not on the allowed list", "line 10: .f_back is not on the allowed list",
        "line 10: .f_locals is not on the allowed list", "line 11: .gi_frame is not on the allowed list",
        "line 11: .f_builtins is not on the allowed list", "line 12: .__name__ is not on the allowed list"],
    "peek_format_walk": ["line 9: .format is not on the allowed list", "line 10: .format is not on the allowed list"],
    "peek_fullwidth_open": ["line 10: builtin open is not on the allowed list", "line 11: ._d is not on the allowed list",
                            "line 12: __builtins__ starts with an underscore"],
    "peek_builtins_dunder": ["line 8: __builtins__ starts with an underscore", "line 9: __import__ starts with an underscore",
                             "line 10: .listdir is not on the allowed list"],
    "peek_reflection_builtins": [
        "line 9: builtin getattr is not on the allowed list", "line 10: builtin vars is not on the allowed list",
        "line 11: builtin globals is not on the allowed list", "line 12: builtin eval is not on the allowed list",
        "line 13: builtin type is not on the allowed list"],
    "peek_builtins_alias": ["line 7: __builtins__ starts with an underscore"],
    "peek_shadow_then_use": ["line 9: builtin open is not on the allowed list", "line 10: builtin open is not on the allowed list",
                             "line 14: builtin open is not on the allowed list"],
    "peek_fstring_attribute": ["line 8: ._d is not on the allowed list"],
    "peek_no_grad_object": ["line 10: .gen is not on the allowed list", "line 10: .gi_frame is not on the allowed list",
                            "line 11: .f_globals is not on the allowed list"],
    "peek_import_tricks": ["line 9: import odometer._data", "line 10: import os", "line 11: from . import",
                           "line 12: from math import sumprod", "line 13: from odometer import *"],
    "peek_docstring": ["line 9: .__doc__ is not on the allowed list"],
    "state_mutable_default": ["line 8: a default argument must be a constant"],
    "state_module_list": ["line 7: module level allows only imports, defs and constants"],
    "state_nested_closure": ["line 17: module level allows only imports, defs and constants"],
    "state_decorator_closure": ["line 23: decorators are not allowed"],
    "state_annotation_walrus": ["line 8: an annotation must be a constant"],
    "state_grad_on_param": ["line 12: assigning to .grad"],
    "state_grad_on_module": ["line 11: assigning to .grad"],
    "state_function_attribute": ["line 9: .seen is not on the allowed list", "line 9: assigning to .seen",
                                 "line 10: assigning to .grad"],
    "state_grad_store_targets": ["line 9: assigning to .grad", "line 11: assigning to .grad", "line 13: assigning to .grad"],
    "state_lambda_default": ["line 7: a default argument must be a constant",
                             "line 7: module level allows only imports, defs and constants"],
    "state_walrus_default": ["line 8: a default argument must be a constant"],
    "budget_try_except": ["line 14: Try is not allowed"],
    "budget_monkeypatch_remaining": ["line 7: assigning to .remaining",
                                     "line 7: module level allows only imports, defs and constants"],
    "budget_monkeypatch_matmul": ["line 12: assigning to .matmul", "line 13: builtin setattr is not on the allowed list"],
    "budget_engine_meter": ["line 12: ._engine is not on the allowed list", "line 12: ._M is not on the allowed list",
                            "line 12: assigning to .budget"],
    "python_free_map": ["line 18: builtin map is not on the allowed list", "line 18: .sumprod is not on the allowed list"],
    "hardcoded_base85_table": ["literals total 5488 bytes, over the cap of 4096"],
    "hardcoded_giant_int": ["literals total 4330 bytes, over the cap of 4096"],
    "hardcoded_many_numbers": ["literals total 8341 bytes, over the cap of 4096"],
    "hardcoded_keyword_names": ["literals total 8440 bytes, over the cap of 4096"],
    "hardcoded_function_name": ["literals total 8433 bytes, over the cap of 4096"],
    "hardcoded_unicode_string": ["literals total 13071 bytes, over the cap of 4096"],
    "hardcoded_name_table": ["source has 33161 syntax nodes, over the cap of 8192"],
    "syntax_deep_nesting": ["syntax error: Parser stack overflowed - Python source too complex to parse"],
}


@pytest.mark.parametrize("name", STATIC)
def test_static_rules_reject(name):
    assert sorted(verify.static_problems(source(name))) == sorted(STATIC[name])


@pytest.mark.parametrize("name, line", [
    ("peek_fullwidth_open", "  line 10: builtin open is not on the allowed list"),
    ("hardcoded_keyword_names", "  literals total 8440 bytes, over the cap of 4096"),
    ("state_decorator_closure", "  line 23: decorators are not allowed"),
])
def test_command_line_exits_2_on_static_failure(name, line, tmp_path):
    cmd = [sys.executable, "verify.py", str(ATTACKS / f"{name}.py"), "--smoke", "--out", str(tmp_path / "x.json")]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 2
    assert proc.stdout.startswith("static rules failed:\n")
    assert line in proc.stdout.splitlines()


# ---------------------------------------------------------------- past the static rules, voided by the child processes

VOIDED = {
    "budget_runs_out": "void: BudgetExhausted escaped train() (budget 2.000e+09 reached at 1.998e+09)",
    "budget_huge_final_op": "void: BudgetExhausted escaped train() (budget 2.000e+09 reached at 2.503e+07)",
    "budget_arange_refund": "void: BudgetExhausted escaped train() (budget 2.000e+09 reached at 1.994e+09)",
    "budget_range_index": "void: rule violation: indexing takes ints, slices, None and ...; use embedding() to gather",
    "item_flood": "void: rule violation: more than 20000 item() reads",
    "nan_logits": "void: rule violation: forward() returned non-finite logits",
    "return_non_tensor": "void: train() must return a dict of name -> Tensor",
    "return_not_dict": "void: train() must return a dict of name -> Tensor",
    "return_int_tensor": "void: parameters must be float tensors",
    "return_over_cap": OVER_CAP,
    "param_name_channel": OVER_CAP,
    "forward_over_cap": "void: forward() cost more than 50,000 ticks per byte",
    "forward_wrong_shape": BAD_LOGITS,
    "forward_non_tensor": BAD_LOGITS,
    "forward_int_logits": BAD_LOGITS,
    "eval_backward": "void: rule violation: backward() is not allowed during evaluation",
    "eval_randn": "void: rule violation: randn() is not allowed during evaluation",
    "peek_item_in_forward": "void: rule violation: item() is not allowed during evaluation",
    "peek_percent_format": "void: rule violation: use item() to read a scalar",
    "peek_branch_on_value": "void: rule violation: a tensor has no truth value; use item()",
    "state_update_in_forward": "void: rule violation: update() is not allowed during evaluation",
    "python_while_true_train": LINE_CAP,
    "python_while_true_forward": LINE_CAP,
    "python_comprehension": LINE_CAP,
    "python_sorted_key": LINE_CAP,
}


@pytest.mark.parametrize("name", VOIDED)
def test_void_through_real_children(name, tmp_path):
    proc, seed, _ = smoke(name, tmp_path)
    assert seed["status"] == VOIDED[name]
    assert proc.returncode == 1
    assert "void: at least one seed failed, so the entry has no score" in proc.stdout


@pytest.mark.parametrize("name, line", [
    ("peek_kwargs_expansion", "TypeError: embedding() got an unexpected keyword argument '_d'"),
    ("peek_print_file", "AttributeError: module 'odometer' has no attribute 'write'"),
])
def test_crash_in_eval_voids_the_run(name, line, tmp_path):
    proc, seed, logs = smoke(name, tmp_path)
    assert seed["status"] == f"void: eval process crashed (see seed-{SEED}-eval.log)"
    assert proc.returncode == 1
    assert line in (logs / f"seed-{SEED}-eval.log").read_text().splitlines()


def test_huge_final_op_is_refused_before_the_work(tmp_path):
    _, seed, _ = smoke("budget_huge_final_op", tmp_path)
    assert seed["train_seconds"] < 5  # the product itself is 8e12 multiplies, many seconds on any machine


def test_python_loops_void_through_the_counter_not_the_wall_limit(tmp_path):
    _, seed, _ = smoke("python_while_true_train", tmp_path)
    assert seed["train_python_lines"] == 20_000_001
    assert seed["train_ticks"] == 0
    assert seed["train_seconds"] < verify.SMOKE["wall_limit_seconds"]


# ---------------------------------------------------------------- caught by the twins, and worth catching

@pytest.mark.parametrize("name, least", [("noncausal_next_byte", 1.0), ("crossrow_batch_mean", 1e-2),
                                         ("noncausal_sub_tolerance", 5e-4)])
def test_twins_catch_a_forward_that_passes_every_static_rule(name, least, tmp_path):
    assert verify.static_problems(source(name)) == []
    proc, seed, _ = smoke(name, tmp_path)
    assert re.fullmatch(TWIN_FAILED, seed["status"])
    assert seed["twin_max_diff"] > least
    assert proc.returncode == 1


def test_read_ahead_would_score_near_zero_without_the_twins(tmp_path):
    seed = children("noncausal_next_byte", tmp_path, twin_tolerance=float("inf"))
    assert seed["status"] == "ok"
    assert seed["bpb"] < 0.2


def test_one_percent_read_ahead_is_caught_at_full_window_count(tmp_path):
    # SMOKE's 8 twins would see it only about one seed in six, so this runs the real track's 256
    assert verify.static_problems(source("noncausal_one_percent")) == []
    seed = children("noncausal_one_percent", tmp_path, eval_windows=1024)
    assert re.fullmatch(TWIN_FAILED, seed["status"])


def test_one_percent_read_ahead_is_a_real_gain(tmp_path):
    honest = children("state_params_dict", tmp_path, eval_windows=1024)
    cheat = children("noncausal_one_percent", tmp_path, eval_windows=1024, twin_tolerance=float("inf"))
    assert cheat["status"] == "ok"
    assert honest["bpb"] - cheat["bpb"] > 0.02


def test_sub_tolerance_bonus_is_a_real_gain(tmp_path):
    honest = children("state_params_dict", tmp_path)
    cheat = children("noncausal_sub_tolerance", tmp_path, twin_tolerance=float("inf"))
    assert honest["bpb"] - cheat["bpb"] > 5e-4


# ---------------------------------------------------------------- runs, and gains nothing

def test_honest_training_price():
    # 40 steps of (batch, embedding, cross_entropy, backward, mul, sub, update), then the 256 x 256 table setup
    batch = 2 * 2000 + 64 * 16 + 16 * 64
    embedding = 2000 + 8 * 1024 + 1024 * 256 // 4
    cross_entropy = 2 * 2000 + 2 * 1024 * 256
    backward = (2 * 2000 + 1024 * 256) + (2 * 2000 + 16 * 1024 * 256 + 65536 // 16)
    step = batch + embedding + cross_entropy + backward + (2000 + 65536 // 4) * 3
    setup = (2000 + 65536 // 16) + (2000 + 65536 // 4)
    assert setup + 40 * step == HONEST_TICKS


def test_state_in_the_params_dict_does_not_survive(tmp_path):
    proc, seed, logs = smoke("state_params_dict", tmp_path)
    assert seed["status"] == "ok"
    assert seed["twin_max_diff"] == 0.0
    assert seed["train_ticks"] == HONEST_TICKS
    assert "STATE SURVIVED" not in (logs / f"seed-{SEED}-eval.log").read_text()
    assert proc.returncode == 0


def test_fstring_conversions_print_the_shape_and_no_values(tmp_path):
    _, seed, logs = smoke("peek_fstring_conversions", tmp_path)
    assert seed["status"] == "ok"
    row = "Tensor(shape=(16, 512), int)"
    last = "Tensor(shape=(8, 512), int)"
    assert (logs / f"seed-{SEED}-eval.log").read_text().splitlines() == ["|".join([row] * 4)] * 2 + ["|".join([last] * 4)]


def test_surrogate_print_cannot_crash_the_verifier(tmp_path):
    proc, seed, _ = smoke("crash_surrogate_print", tmp_path)
    assert seed["status"] == "ok"
    assert proc.returncode == 0


def test_param_copy_is_charged(tmp_path):
    _, seed, _ = smoke("budget_param_free_copy", tmp_path)
    big = 2000 + (1 << 20) // 16
    copies = 100 * (2000 + (1 << 20) // 4)
    table = (2000 + 65536 // 16) + (2000 + 65536 // 4)
    assert seed["train_ticks"] == big + copies + table == 26_506_416


def test_thin_matmul_costs_what_a_multiply_costs(tmp_path):
    _, seed, _ = smoke("budget_thin_matmul", tmp_path)
    setup = (2000 + (1 << 20) // 16) + 2000
    products = 10 * (2000 + (1 << 20) * 8 * 64 // 128 + 256)
    table = (2000 + 65536 // 16) + (2000 + 65536 // 4)
    assert seed["train_ticks"] == setup + products + table == 42_059_616


# ---------------------------------------------------------------- the control

def test_honest_entry_passes_end_to_end(tmp_path):
    proc, seed, _ = smoke("legit_tiny", tmp_path)
    assert proc.returncode == 0
    assert seed["status"] == "ok"
    assert seed["train_ticks"] == 513_400_480
    assert seed["twin_max_diff"] == 0.0
    assert seed["params"] == 65536
    assert 3.0 < seed["bpb"] < 4.5  # far below the 8.0 of a uniform guess, and above what a bigram can do on any machine


def test_reference_entry_still_passes_the_static_rules():
    src = (ROOT / "entries" / "gpt" / "train.py").read_text(encoding="utf-8")
    assert verify.static_problems(src) == []


# ---------------------------------------------------------------- known open: these pin what still works

def test_open_c_level_loop_is_stopped_only_by_the_wall_limit(tmp_path):
    seed = children("open_c_level_sum", tmp_path, wall_limit_seconds=2)
    assert seed["status"] == "void: train process passed the wall limit"


def test_open_big_integers_are_free(tmp_path):
    seed = children("open_bigint_vector", tmp_path)
    assert seed["status"] == "ok"
    assert seed["train_python_lines"] == 22
    assert seed["train_ticks"] == 24_480


def test_open_long_line_counts_once(tmp_path):
    seed = children("open_long_line", tmp_path)
    assert seed["status"] == "ok"
    assert seed["train_python_lines"] == 3008  # 1000 loop passes, each with 2000 additions the counter does not see


def test_open_below_tolerance_bonus_buys_almost_nothing(tmp_path):
    honest = children("state_params_dict", tmp_path)
    cheat = children("open_below_tolerance", tmp_path)
    assert cheat["status"] == "ok"
    assert cheat["twin_max_diff"] < verify.TRACK["twin_tolerance"]
    assert 0 < honest["bpb"] - cheat["bpb"] < 1.5e-4 < verify.RECORD_MARGIN
