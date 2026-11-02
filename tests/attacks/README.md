# Attacks

Each file is an entry written to cheat. `tests/test_attacks.py` runs every one and pins the exact rejection. `legit_tiny` is the control and must pass.

"Hole" means the attack got further than it should on the code before this suite. The fix is named beside it. Numbers are `--smoke` at seed 11 on an M-series Mac unless noted.

## Reading what it should not

- `peek_open`, `peek_numpy`, `peek_odometer_data`, `peek_import_tricks`, `peek_builtins_dunder`, `peek_reflection_builtins`: static rule on the name.
- `peek_fullwidth_open`: fullwidth `open` folds to plain `open` in the parser, so the rule sees through it.
- `peek_generator_frame`, `peek_no_grad_object`, `peek_format_walk`, `peek_fstring_attribute`, `peek_docstring`, `peek_match_pattern`: static rule on the attribute or the node.
- `peek_shadow_then_use`: hole. Rebinding `open` as a local hid the name from the rule, and only a NameError in the child stopped it. Any use of a forbidden name now fails, in any scope.
- `peek_builtins_alias`: hole. `import odometer as __builtins__` passed the rule and swapped the table free names resolve in. Import aliases now follow the underscore rule.
- `peek_kwargs_expansion`, `peek_print_file`: past the rules, the child crashes and the run is void.
- `peek_percent_format`, `peek_branch_on_value`, `peek_item_in_forward`: a tensor refuses to become a Python value in `forward()`.
- `peek_fstring_conversions`: runs, and the log shows only shapes.

## Looking ahead or across the batch

- `noncausal_next_byte`, `crossrow_batch_mean`: twin check. Without the twins read-ahead scores under 0.2 bpb, which the test measures.
- `noncausal_one_percent`: reads ahead in 1% of rows and gains over 0.02 bpb. Smoke has 8 twins and missed it at seed 11. At the real 1024 windows it was void on 7 of seeds 11 to 18, and one void seed voids the entry.
- `noncausal_sub_tolerance`: hole. A 5e-4 bonus on the true next byte moved twins by 7e-4 nats and passed, for about 6e-4 bpb. `twin_tolerance` is now 1e-4.

## Keeping state between calls

- `state_module_list`, `state_mutable_default`, `state_lambda_default`, `state_walrus_default`, `state_nested_closure`, `state_function_attribute`: static rules.
- `state_decorator_closure`, `state_annotation_walrus`: hole. Both run at def time and built state that survived, so only the twins stopped them, at 4.78 and 0.11 nats. Decorators are banned and annotations must be constants.
- `state_grad_on_module`, `state_grad_on_param`, `state_grad_store_targets`: hole. Assigning `.grad` passed the rule on any object, and `od.grad = []` persisted. Only `x.grad = None` is allowed now.
- `state_params_dict`: runs and gains nothing, a new dict per call.
- `state_update_in_forward`: `update()` is train only.

## Getting compute free

- `budget_arange_refund`: hole. `od.arange(-10**15)` had a negative price, so ticks reached -2.5e14 with status ok. `_tick` clamps at zero.
- `budget_param_free_copy`: hole. `param()` copied for free. It now pays per element.
- `budget_thin_matmul`: hole. A thin inner dimension ran at 87 times the median nanoseconds per tick. The inner dimension now counts as at least 64.
- `budget_huge_final_op`: hole. One product ran 12 s before the meter refused it. matmul now charges before the work.
- `budget_range_index`: hole. `x[range(n)]` was a fancy index at the price of a view. Indexing keys are now an allowlist.
- `budget_runs_out`, `budget_try_except`, `budget_monkeypatch_remaining`, `budget_monkeypatch_matmul`, `budget_engine_meter`: void or a static rule.
- `python_free_map`: hole. `map(math.sumprod, ...)` was a matmul in 10 counted lines and 36768 ticks. `map` and the sequence and big-integer math functions are gone.
- `python_while_true_train`, `python_while_true_forward`, `python_comprehension`, `python_sorted_key`, `item_flood`, `forward_over_cap`, `eval_backward`, `eval_randn`: the line counter, the item ration, the per-byte cap, or a train-only rule.

## Shipping data in the source

- `hardcoded_base85_table`, `hardcoded_giant_int`, `hardcoded_many_numbers`: literal cap.
- `hardcoded_keyword_names`, `hardcoded_function_name`, `hardcoded_unicode_string`, `hardcoded_name_table`: hole. All four passed with a table in the source and scored 3.26 bpb with no training step, against 3.87 for 40 honest steps. The cap now counts UTF-8 bytes, keyword names and function names, and a new cap of 8192 syntax nodes stops tables spelled as variable names.
- `param_name_channel`: hole. Parameter names carried data from train to eval for free. Each character now counts as one parameter.

## Breaking the verifier or the contract

- `crash_surrogate_print`: hole. A surrogate in stdout raised UnicodeDecodeError in the parent, with no verdict and no log. Child output now goes straight to the log file.
- `syntax_deep_nesting`: hole. The parser's MemoryError reached the parent. It is now a syntax error.
- `return_not_dict`, `return_non_tensor`, `return_int_tensor`, `return_over_cap`, `forward_wrong_shape`, `forward_non_tensor`, `forward_int_logits`, `nan_logits`: void with an exact status.

## Still open

These still work. The tests pin each one, so a change in either direction shows up.

- `open_c_level_sum`: `sum(range(n))` is one counted line. Only the wall limit stops it. Shorter loops are free compute.
- `open_bigint_vector`: Python big integers do vector arithmetic with almost no ticks or lines.
- `open_long_line`: a line with 2000 additions counts once.
- `open_below_tolerance`: a bonus just under the twin tolerance passes. The gain is under 1.5e-4 bpb, 13 times below the 0.002 record margin.

Free compute in the first three is limited by data, since `item()` rations train bytes to 20000 reads and tensors cannot reach Python any other way. That the limit removes any score gain is inferred, not measured.

Not tested: memory and disk exhaustion have no cap. A smuggled table of 4 to 8 KB remains possible, the same order as the literal cap. Twin noise reached 1.8e-5 across 944 stored seed runs, against the tolerance of 1e-4.
