# The reviews' briefs

Every review of a record, and the final review, was written by a fresh Claude session with no memory of the work, started with the brief below. The review transcripts are not in this repo, because they carry local paths. What each one wrote is in `records/<name>/review.md` or `study/attempts/<id>/claims/<n>/review.md`. A review refuses a record with the last line `Verdict: not real`. None has done so yet.

## Fresh review of the five recipes

Model: Claude Opus 5.5.

```text
You are a fresh, skeptical reviewer with no stake in this project. Repo: the repo root. It is a speedrun track: train a byte-level language model on TinyStories under a fixed budget of metered ops (verify.py holds the rules; odometer/_engine.py the prices; lower held-out bits per byte wins). Entries are in entries/{gpt,rnn,modern,ppm,mixer}/train.py with dev numbers in entries/TUNING.md (summary table near the top; you do not need to read the whole file). Do not run any full-budget training; `--smoke --split dev` is allowed for checks.

For each of the five entries, in this ladder order (gpt, rnn, modern, ppm, mixer), write a review file records-reviews/<name>.md (create the directory) that answers, in plain sentences, under 180 words each: 1) what actually drives its number: the mechanism, tied to specific lines and to the price list, not a restatement of the docstring; 2) why it improves on the previous row in the ladder, if it does (this is the question that matters: is the improvement real and explainable, or an artifact of the rules?); 3) does it abuse a rule, a mispriced op or a loophole in the track? Look hard, especially at the mixer's window_votes and PPM backoff at eval and the ppm entry's count tables, including whether their eval price is what it should be and whether anything computed at eval escapes the cap; 4) verdict line: `Verdict: real` or `Verdict: artifact: <why>`. Also state one thing you could not check. Do not state anything you did not verify by reading the code or running it. No em dashes. End your final message with the five verdicts and, if any, the single most serious concern in under 100 words. Do not commit.
```

## Fresh review of pilot-02 claim

Model: Claude Opus 5.5.

```text
You are a fresh, skeptical reviewer for a training speedrun called odometer, repo at the repo root. Nobody who built it is in the room. An agent entry passed the statistical record rule. A record on this table counts only if you can explain why the gain is real. Your job is to try to break that claim and write the review. If it is an exploit, say so. A "not real" verdict is a perfectly good outcome.

THE TRACK
- `verify.py` is the referee (docstring at the top, static rules, child train and child eval processes, judge). `odometer/_engine.py` holds the price list (top of file) and metering. `odometer/_data.py` holds the data access.
- Score: held-out bits per byte on TinyStories. Budget: 4e10 ticks for everything train() does. A tick is priced from shapes only and is meant to track time, about one nanosecond of a laptop core. `tools/calibrate.py` checks each op's ns per tick stays within 5x of the median.
- `tests/attacks/README.md` lists known attacks and holes that remain open.

THE ENTRY
- Claim under review: `study/attempts/pilot-02/claims/2/train.py`, with `claim.json`, `verify.json`, `verify.log`, and per-seed logs under `verify/` next to it. Claim 1 is the same code with a different table schedule.
- Incumbent: `records/05-mixer/train.py`, its `review.md`, and `log.json`.
- The agent's whole session is in `study/attempts/pilot-02/transcript.jsonl` (stream-json). Its working files are in `study/attempts/pilot-02/work.tar.gz`. Extract them to a temp directory if you need them, never into the repo.

MEASURED ALREADY, ALL IN GITHUB ACTIONS (read the logs, do not take this list on trust)
- Incumbent dev mean 1.1566 (records/05-mixer/log.json). Test 1.1555.
- Claim 2 verified on the sealed test split: stage 1 and stage 2 logs in verify.json, about 1.1434.
- Ablations on dev, 8 seeds each: `study/attempts/pilot-02/ablations/fold_only.py` and `fold_only.json` (claim's folded head with the record's uniform every-4 table schedule) and `lazy_only.py` and `lazy_only.json` (record's 256-way head with the claim's lazy schedule).

ANSWER THESE, WITH EVIDENCE
1. Mechanism. What exactly buys the gain? Quantify bytes trained per tick against the incumbent, with numbers you derived. Do the ablations add up, and what does the split tell you?
2. Rare bytes. `candidates()` gets a softmax slot and all other bytes get FLOOR plus window votes at eval. Is that honest compression modelling? How was the list chosen, from train, dev or test? Check the transcript and the data. Could this list or any literal in the file smuggle information about the held-out split? Compute what the list costs or gains on dev, without training, by counting how often non-candidate bytes occur.
3. Price exploitation. Does the lazy schedule or anything else get real work the price list undercharges? Compare the entry's train_seconds per tick and eval_seconds per tick with the incumbent's on the same CI runner class. Accumulating gradients between updates: is that priced? Is the eval side, including the python loop building `place`, within the rules, and what eval_ticks_per_byte does it reach? A legitimate reallocation inside the price list is not an exploit, but if the prices are wrong in a way that lets real work go uncounted, say so.
4. Rule abuse. Twin check, fresh params, static rules, python lines, item reads, memory. Any look-ahead or leak? Does it depend on the seed in an odd way? The record rule raises a flag when seed SD is far below the incumbent's; check whether that applies.
5. Selection and noise. How much did the agent's own process select on seeds, and does the verified gain on fresh seeds on the sealed split stand clear of that? Did it see any test data? Its copy had test windows zeroed. Check it did not try to rebuild the split.
6. Claim 1 against claim 2: is there a real difference, and does it matter which is seated?

LIMITS
- This Mac is under load and another job holds its training lock. Do not run training on it: no `verify.py` at the real budget, no long jobs. Reading files, small python for counting bytes in `data/`, and pricing a handful of ops are fine. `.venv/bin/python` has the dependencies.
- Do not edit anything in the repo except the one file below, and do not commit or push. No web use is needed.

OUTPUT
Write your review to `study/attempts/pilot-02/claims/2/review.md`. Plain prose in numbered points matching the six questions, each with the numbers you found, then a line starting `Could not check:`, and a final line that is exactly `Verdict: real` or `Verdict: not real`. Keep it under 45 lines. Do not use em dashes anywhere. Every number must be one you read or computed, never one copied from this brief without checking.

Then reply with: the verdict, the two or three findings that matter most, and anything that should be disclosed publicly about this record even if the verdict is real.
```

## Fresh review of attempt-01 claim

Model: Claude Opus 5.5.

```text
You are a fresh, skeptical reviewer for a training speedrun called odometer, repo at the repo root. Nobody who built it is in the room. An agent entry passed the statistical record rule. A record on this table counts only if you can explain why the gain is real. Your job is to try to break that claim and write the review. If it is an exploit, say so. A "not real" verdict is a perfectly good outcome.

THE TRACK
- `verify.py` is the referee (docstring at the top, static rules, child train and child eval processes, judge). `odometer/_engine.py` holds the price list (top of file) and metering. `odometer/_data.py` holds the data access, including how held-out windows are cut.
- Score: held-out bits per byte on TinyStories. Budget: 4e10 ticks for everything train() does. A tick is priced from shapes only and is meant to track time. `tests/attacks/README.md` lists known attacks and holes that remain open.

THE ENTRY
- Claim under review: `study/attempts/attempt-01/claims/1/train.py`, with `claim.json`, `verify.json`, `verify.log`, `outcome.json` and per-seed logs under `verify/` next to it.
- Incumbent: `records/06-mixer-fold/train.py`, its `review.md` (an earlier reviewer's work, read it for format only and reach your own view), and `log.json`.
- The agent's whole session is in `study/attempts/attempt-01/transcript.jsonl` (stream-json). Its working files, including its own experiment logs, are in `study/attempts/attempt-01/work.tar.gz`. Extract them to a temp directory if you need them, never into the repo.
- The agent's own account of each change is in claim.json. Treat it as a claim to check, not as fact.

ANSWER THESE, WITH EVIDENCE
1. Mechanism. Name each change that buys gain: masking contexts that reach before the window, raw-hash in-window votes with five stages, batch 128 with table schedule, smaller softmax (76 classes), learning rates, table-rate decay. What evidence exists for each one's size? Check the agent's logs against its claims, and quantify bytes trained per tick against the incumbent.
2. The cold-start fix is the most interesting part. A window of 512 bytes starts mid-story, and the verifier feeds forward() no bytes from before it. Is masking those contexts honest modelling of the task the track sets, or does it exploit how windows are cut? Could an entry get a score the track did not intend from it? Would the gain carry over to any real use of such a model, or only to this window format? Say so either way, since it should be disclosed.
3. The vote stages. Five (order, concentration, discount) constants and a byte list were fitted by the agent. Were they fitted on dev? Does the gain survive on test (it does in verify.json, check it)? Do the 15 constants or any literals carry information about held-out data beyond ordinary tuning? How many variants did the agent compare, and on how many seeds?
4. Rare bytes. `candidates()` names a smaller set than the incumbent's. Count, from `data/`, without training: how often held-out dev bytes fall outside it, and what that costs at FLOOR. Which file was the list chosen from? Check the transcript. Confirm no tool call read test windows (they are zeroed in the agent's copy; `sandbox.py` makes that copy) and that it did not try to rebuild the split from the web.
5. Price exploitation. train_seconds is about 59 per seed on a 2-core runner, against about 90 for the incumbent, for the same ticks. Why is the same tick count so much cheaper in wall time? Does that mean the record is overcharged for something, or that this entry has moved to ops the list overprices? Conversely, does anything get real work the list undercharges? eval_ticks_per_byte is about 17,800 against a cap of 50,000: is that within the rules and is the python loop building `place` and `history_masks` priced? A legitimate reallocation inside the price list is not an exploit.
6. Rule abuse. Twin check (twin_max_diff is 0.0 on the seed I looked at; is that expected, and could masks defeat the check?), fresh params, static rules, python lines, item reads, memory. Any look-ahead or cross-row leak? The record rule raises a flag when seed SD is far below the incumbent's (read `verify.py` for LOW_VARIANCE_RATIO); check whether that fired in `verify.json`, and if so whether it points at a fixed init or data order.
7. Selection. Does the verified gain on fresh seeds on the sealed split stand clear of the agent's selection? Note the claimed gain was 0.0184 and the verified gain 0.0185.

LIMITS
- This Mac is under load and another job holds its training lock. Do not run training on it: no `verify.py` at the real budget, no long jobs. Reading files, small python for counting bytes in `data/`, and pricing a handful of steps are fine, seconds in total. `.venv/bin/python` has the dependencies.
- Do not edit anything in the repo except the one file below, and do not commit or push. No web use is needed. Be economical: you do not need to read the whole transcript, search it.

OUTPUT
Write your review to `study/attempts/attempt-01/claims/1/review.md`. Plain prose in numbered points matching the seven questions, each with the numbers you found, then a line starting `Could not check:`, and a final line that is exactly `Verdict: real` or `Verdict: not real`. Keep it under 45 lines. Do not use em dashes anywhere. Every number must be one you read or computed, never one copied from this brief without checking.

Then reply with: the verdict, the two or three findings that matter most, and anything that should be disclosed publicly about this record even if the verdict is real.
```

## Fresh review of attempt-02 claim

Model: Claude Opus 5.5.

```text
You are a fresh, skeptical reviewer for a training speedrun called odometer, repo at the repo root. Nobody who built it is in the room. An agent entry passed the statistical record rule. A record counts only if you can explain why the gain is real. Try to break that claim and write the review. If it is an exploit, say so. "Not real" is a perfectly good outcome.

TRACK: `verify.py` is the referee, `odometer/_engine.py` holds the price list and metering, `odometer/_data.py` the data access and window cutting. Score: held-out bits per byte on TinyStories. Budget: 4e10 ticks for all of train(). `tests/attacks/README.md` lists known attacks and open holes.

ENTRY: `study/attempts/attempt-02/claims/1/train.py` with `claim.json`, `verify.json`, `outcome.json`, and seed logs in `verify/` beside it. Incumbent: `records/07-mixer-cold/` (train.py, log.json, and review.md, which you may read for format only; reach your own view). The agent's session is `study/attempts/attempt-02/transcript.jsonl`, its working files and experiment logs are in `work.tar.gz` beside it (extract to a temp directory, never into the repo). claim.json is the agent's account; treat it as a claim to check.

The claim, in short: tables train on 92% of the budget, then the last 8% fits a small gate with the tables frozen. The gate replaces hand-set in-window PPM constants with functions of position features, adds an order-1 in-window stage, and mixes softmax and last stage as logits. Votes run over 76 classes.

ANSWER WITH EVIDENCE:
1. Mechanism. What buys the gain, how much each part, and how do you know? Is this second-stage mixing (compare SSE/APM in PAQ) doing real modelling? Bytes trained per tick against the incumbent, derived by you.
2. Honesty of the gate. It is fit in train() with frozen tables on training bytes. The training bytes were also used to train the tables, so the gate sees tables already fitted on that data: does the gate then miscalibrate, and does held-out performance show it? Does anything in it read targets, look ahead, or hold state across calls at eval? Twin check values in the logs: real differences or zero? Could the gate or its features defeat the twin check?
3. Dev tuning. Stage orders and gate settings were chosen on a dev subset. How many variants, how many seeds? Do literals or constants carry held-out information beyond ordinary tuning? Did the agent read any test window, or try to rebuild the split from the web? (`study/sandbox.py` zeros test windows in its copy.) The claimed gain was 0.0172 and the verified gain 0.0188, so check whether that gap in the favourable direction points at anything.
4. Price exploitation. Does anything get real work the list undercharges or dodge the meter? Eval ticks per byte against the 50,000 cap; python lines; item reads against the ration (does the gate use item()?); memory; wall seconds per tick against the incumbent on comparable hosts (`records/07-mixer-cold/test.json`). Seed SD against LOW_VARIANCE_RATIO in `verify.py`.
5. Selection. Does the verified gain on the sealed split stand clear of selection? Use verify.json, both stages.

LIMITS: The Mac is under load and another job holds its training lock. No `verify.py` at the real budget and no long jobs. Reading files, small python counting, and pricing a handful of steps are fine, seconds in total. `.venv/bin/python` has the dependencies. Edit nothing in the repo except the one output file, no commits, no web. Be economical: search the transcript, do not read it through.

OUTPUT: write `study/attempts/attempt-02/claims/1/review.md`: plain prose in numbered points matching the five questions with the numbers you found, then a line starting `Could not check:`, and a final line exactly `Verdict: real` or `Verdict: not real`. Under 40 lines. No em dashes. Every number must be one you read or computed.

Reply with: the verdict, the two or three findings that matter most, and anything to disclose publicly about this record even if real.
```

## Fresh review of attempt-04 claim

Model: Claude Opus 5.5.

```text
You are a fresh, skeptical reviewer for a training speedrun called odometer, repo at the repo root. Nobody who built it is in the room. An agent entry passed the statistical record rule. A record counts only if you can explain why the gain is real. Try to break that claim and write the review. If it is an exploit, say so. "Not real" is a perfectly good outcome.

TRACK: `verify.py` is the referee, `odometer/_engine.py` holds the price list and metering, `odometer/_data.py` the data access and window cutting. Score: held-out bits per byte on TinyStories. Budget: 4e10 ticks for all of train(). `tests/attacks/README.md` lists known attacks and open holes.

ENTRY: `study/attempts/attempt-04/claims/1/train.py` with `claim.json`, `verify.json`, `outcome.json`, and seed logs in `verify/` beside it. Incumbent: `records/08-mixer-gate/` (train.py, log.json, and review.md, which you may read for format only; reach your own view). The agent's session is `study/attempts/attempt-04/transcript.jsonl`, its working files and experiment logs are in `work.tar.gz` beside it (extract to a new temp directory, never into the repo). claim.json is the agent's account; treat it as a claim to check.

The claim, in short: low-order tables pay embedding backward on only a fraction of steps, two new in-window PPM stages are keyed on the case-folded word prefix and on prefix plus previous word, and votes from before the last story boundary in the window count 0.1.

ANSWER WITH EVIDENCE:
1. Mechanism. What buys the gain, how much each part, and how do you know? Bytes trained per tick against the incumbent, derived by you. Note an earlier attempt (`study/attempts/attempt-03/`) tried a similar low-order schedule and gained 0.0025 without a record; how much of this gain is that same effect and how much is the new word stages and boundary weighting?
2. Honesty. Do word-prefix stages or the story-boundary weighting read targets, look ahead, or hold state across calls at eval? Is the NUL story separator in the window a legitimate signal the track allows, and could the boundary weighting exploit how windows are cut? Twin check values in the logs: real differences or zero? Could these features defeat the twin check?
3. Dev tuning. How many variants, how many seeds (`seed_runs_before` is 44, and earlier reviews found this undercounts when lab scripts call verify's children directly)? Do constants carry held-out information beyond ordinary tuning? Did the agent read any test window, or try to rebuild the split from the web? (`study/sandbox.py` zeros test windows in its copy.)
4. Price exploitation. Does anything get real work the list undercharges or dodge the meter? Eval ticks per byte against the 50,000 cap; python lines; item reads; memory; wall seconds per tick against the incumbent on comparable hosts (`records/08-mixer-gate/test.json`). An earlier review found the gate phase ran 5.4 to 5.9 times slower per tick than the table phase on x86: does this entry lean harder on that? Seed SD against LOW_VARIANCE_RATIO in `verify.py`.
5. Selection. Does the verified gain on the sealed split stand clear of selection? Use verify.json, both stages.

LIMITS: The Mac is under load and another job holds its training lock. No `verify.py` at the real budget and no long jobs. Reading files, small python counting, and pricing a handful of steps are fine, seconds in total. `.venv/bin/python` has the dependencies. Edit nothing in the repo except the one output file, no commits, no web. Be economical: search the transcript, do not read it through.

OUTPUT: write `study/attempts/attempt-04/claims/1/review.md`: plain prose in numbered points matching the five questions with the numbers you found, then a line starting `Could not check:`, and a final line exactly `Verdict: real` or `Verdict: not real`. Under 40 lines. No em dashes. Every number must be one you read or computed.

Reply in under 200 words: the verdict, the two or three findings that matter most, and anything to disclose publicly about this record even if real.
```

## Fresh review of attempt-05 claim

Model: Claude Opus 5.5.

```text
You are a fresh, skeptical reviewer for a training speedrun called odometer, repo at the repo root. Nobody who built it is in the room. An agent entry passed the statistical record rule. A record counts only if you can explain why the gain is real. Try to break that claim and write the review. If it is an exploit, say so. "Not real" is a perfectly good outcome.

TRACK: `verify.py` is the referee, `odometer/_engine.py` holds the price list and metering, `odometer/_data.py` the data access and window cutting. Score: held-out bits per byte on TinyStories. Budget: 4e10 ticks for all of train(). `tests/attacks/README.md` lists known attacks and open holes.

ENTRY: `study/attempts/attempt-05/claims/1/train.py` with `claim.json`, `verify.json`, `outcome.json`, and seed logs in `verify/` beside it. Incumbent: `records/09-mixer-words/` (train.py, log.json, and review.md, which you may read for format only; reach your own view). The agent's session is `study/attempts/attempt-05/transcript.jsonl`, its working files and experiment logs are in `work.tar.gz` beside it (extract to a new temp directory, never into the repo). claim.json is the agent's account; treat it as a claim to check.

The claim, in short: the table model and in-window stages are the incumbent's apart from 6,012 fewer table rows. The gate is fitted differently: one gradient step per gate batch instead of three, its MLP removed, its previous-byte table joined by a table on a hash of the last two bytes, both trained by AdaGrad. The agent says cached-table experiments showed the gate is short of fresh bytes, not gradient steps. It says about 62 variants were screened on dev, mostly on cached tables on seeds 11 and 12, and warns the sealed score may come out above the dev figure.

ANSWER WITH EVIDENCE:
1. Mechanism. What buys the gain, how much each part, and how do you know? Bytes trained per tick against the incumbent, derived by you, for both the table phase and the gate phase. Does removing the MLP and taking fewer steps per batch account for it, or does the second gate table? Are the agent's own ablation numbers supported by its logs?
2. Honesty. Do the gate tables read targets, look ahead, or hold state across calls at eval? Twin check values in the logs: real differences or zero? Could these features defeat the twin check? Does a hash of the last two bytes at each position stay causal at the window's first bytes?
3. Dev tuning. How many variants, how many seeds (`seed_runs_before` is 16, and earlier reviews found this undercounts when lab scripts call verify's children directly; the agent says about 62 variants were screened with cached tables, which that counter cannot see)? Cached-table screening on two seeds is a stronger selection than usual: claimed gain 0.00549 against verified 0.00472, so how much of the difference is selection? Do constants carry held-out information beyond ordinary tuning? Did the agent read any test window, or try to rebuild the split from the web? (`study/sandbox.py` zeros test windows in its copy.)
4. Price exploitation. Does anything get real work the list undercharges or dodge the meter? Eval ticks per byte against the 50,000 cap; python lines; item reads; memory; parameter count against its cap, since the entry gave up rows to fit a new table; wall seconds per tick against the incumbent on comparable hosts (`records/09-mixer-words/test.json`). Earlier reviews found the gate phase ran 5.4 to 5.9 times slower per tick than the table phase on x86: does this entry change that? Seed SD against LOW_VARIANCE_RATIO in `verify.py`.
5. Selection. Does the verified gain on the sealed split stand clear of selection? Use verify.json, both stages.

LIMITS: The Mac is under load and another job holds its training lock. No `verify.py` at the real budget and no long jobs. Reading files, small python counting, and pricing a handful of steps are fine, seconds in total. `.venv/bin/python` has the dependencies. Edit nothing in the repo except the one output file, no commits, no web. Be economical: search the transcript, do not read it through.

OUTPUT: write `study/attempts/attempt-05/claims/1/review.md`: plain prose in numbered points matching the five questions with the numbers you found, then a line starting `Could not check:`, and a final line exactly `Verdict: real` or `Verdict: not real`. Under 40 lines. No em dashes. Every number must be one you read or computed.

Reply in under 200 words: the verdict, the two or three findings that matter most, and anything to disclose publicly about this record even if real.
```

## Fresh review of attempt-06 claim

Model: Claude Opus 5.5.

```text
You are a fresh, skeptical reviewer for a training speedrun called odometer, repo at the repo root. Nobody who built it is in the room. An agent entry passed the statistical record rule. A record counts only if you can explain why the gain is real. Try to break that claim and write the review. If it is an exploit, say so. "Not real" is a perfectly good outcome.

TRACK: `verify.py` is the referee, `odometer/_engine.py` holds the price list and metering, `odometer/_data.py` the data access and window cutting. Score: held-out bits per byte on TinyStories. Budget: 4e10 ticks for all of train(). `tests/attacks/README.md` lists known attacks and open holes.

ENTRY: `study/attempts/attempt-06/claims/2/train.py` with `claim.json`, `verify.json`, `outcome.json`, and seed logs in `verify/` beside it. Claim 1 of the same attempt (`claims/1/`) replicated but did not reach a record. Incumbent: `records/10-mixer-fresh/` (train.py, log.json, and review.md, which you may read for format only; reach your own view). The agent's session is `study/attempts/attempt-06/transcript.jsonl`, its working files and experiment logs are in `work.tar.gz` beside it (extract to a new temp directory, never into the repo). claim.json is the agent's account; treat it as a claim to check.

The claim, in short: same architecture as the record, with three changes. The tables for orders 8, 10 and 12 go from 8 to 4 floats per row with twice the rows, halving their backward cost and reducing collisions. The gate step is cheaper: in-window byte matches come from one equality matrix multiplied along its diagonals, window constants are built once per fit, and a stage multiplies by a reciprocal rather than dividing a whole row. Only four in-window stages remain (orders 3 and 6 and the two word contexts), paying for more gate steps, with the gate tables' AdaGrad rate at 0.15.

ANSWER WITH EVIDENCE:
1. Mechanism. What buys the gain, how much each part, and how do you know? Bytes trained per tick against the incumbent, derived by you, for both the table phase and the gate phase. How much of the gain is thinner high-order tables, how much is more gate steps, how much is the changed stages? Does a shared equality matrix multiplied along its diagonals price what it really does, given the engine's matmul and elementwise prices?
2. Honesty. Do the diagonal matches or any stage read targets, look ahead, or hold state across calls at eval? Is the equality matrix strictly causal at every position? Twin check values in the logs: real differences or zero? Could these features defeat the twin check?
3. Dev tuning. How many variants, how many seeds (`seed_runs_before` is 46, and earlier reviews found this undercounts when lab scripts call verify's children directly or an offline harness scores cached tables)? The agent says most selection was on one seed in an offline harness where differences under about 0.0005 are noise. How much selection does the 0.0041 verified gain reflect? Do constants carry held-out information beyond ordinary tuning? Did the agent read any test window, or try to rebuild the split from the web? (`study/sandbox.py` zeros test windows in its copy.)
4. Price exploitation. Does anything get real work the list undercharges or dodge the meter? Is a matmul used as a cheap way of doing work the list prices higher elsewhere, and is it paying for the real cost? Eval ticks per byte against the 50,000 cap; python lines; item reads; memory; parameter count against its cap; wall seconds per tick against the incumbent on comparable hosts (`records/10-mixer-fresh/test.json`). Earlier reviews found the gate phase ran 5.4 to 5.9 times slower per tick than the table phase on x86: does this entry change that? Seed SD against LOW_VARIANCE_RATIO in `verify.py`.
5. Selection. Does the verified gain on the sealed split stand clear of selection? Use verify.json, both stages. Two claims from one attempt went through the rule; say what that does to the reading of this one.

LIMITS: The Mac is under load and another job holds its training lock. No `verify.py` at the real budget and no long jobs. Reading files, small python counting, and pricing a handful of steps are fine, seconds in total. `.venv/bin/python` has the dependencies. Edit nothing in the repo except the one output file, no commits, no web. Be economical: search the transcript, do not read it through.

OUTPUT: write `study/attempts/attempt-06/claims/2/review.md`: plain prose in numbered points matching the five questions with the numbers you found, then a line starting `Could not check:`, and a final line exactly `Verdict: real` or `Verdict: not real`. Under 40 lines. No em dashes. Every number must be one you read or computed.

Reply in under 200 words: the verdict, the two or three findings that matter most, and anything to disclose publicly about this record even if real.
```

## Final adversarial review before publishing

Model: Claude Opus 5.5.

```text
You are a fresh, hostile reviewer: a senior ML researcher deciding whether odometer, a repo at the repo root about to go public, deserves belief. Nobody who built it is in the room. Find what is wrong, overstated, unsupported or embarrassing before strangers do. A short list of real problems beats a long list of quibbles. Do not soften findings.

WHAT IT CLAIMS, in `README.md`, `docs/copy.json` (the web page's prose), `docs/index.html`, `site/writeup.in.md` (a site write-up), `paper/odometer.tex` (a short paper), `records/names.json` (per-record weak spots):
- Top claim: Claude Code agents set verified records (current: record 11, 1.0887 bits per byte) on a track anyone can rerun, with a preregistered series of 6 attempts: 7 of 7 claims replicated, 5 records.
- Novelty claim: a training budget metered per operation and independent of hardware, for a public record track, was found nowhere else. Survival of agent claims has been measured before, and the docs say so.
- Hardware-free claim: same program, same reading on every machine, scores equal within a tolerance far below the record margin. The final cross-machine matrix cannot run until the repo is public (macOS and arm runners), so `?` placeholders for the fairness table and quickstart time are expected and not findings. Earlier evidence is in `study/fairness/`.

READ: `verify.py`, `odometer/_engine.py` and `_data.py`; `study/PREREG.md`, `study/analyze.py`, `study/rules.py`, `study/results.md`, `study/attempts/*/claims/*/{outcome.json,verify.json,review.md}`, `study/nulls/`; `tests/attacks/README.md`, `review/novelty-and-exploits.md`, `entries/TUNING.md`; `reproduce.py`, `tools/`. Do not read RUNLOG.md or WORKLOG.md or DESIGN.md, which will not be public.

ATTACK, with evidence:
1. Top claim. Pick record 11. Is the path from attempt-06's session to the seated record what the docs say it is, in the logs? Does anything make "verified" weaker than a reader would assume? Is the series clean: was the plan really committed before attempt-01 (use `git log` on study/PREREG.md, dates and contents), and do deviations from it appear under Deviations? Was the count, the model or the prompt changed after seeing outcomes? Are 7 of 7 and the Wilson interval computed the way the plan says? Are claims within attempt-06 treated as the plan says?
2. Does the design make survival nearly automatic? If claims come from agents who can run the verifier on dev with many seeds and the dev and test windows come from one file, is 7 of 7 informative, or is `replicated` (no margin, p<0.01) so easy that it measures little? Does the write-up lean on it harder than that allows? What would a sceptic say the series actually shows?
3. Novelty, with the web. Use at most 12 searches, no fan-out. Look for a prior public training speedrun, leaderboard or benchmark that charges an operation-level or abstract compute budget instead of wall clock, for example deterministic instruction or gas metering of ML training, op-count budgets in NAS or AutoML benchmarks, tick or step-metered challenges, sized MCU or compression contests with computed cost, or a metered training API. Check what `review/novelty-and-exploits.md` did and missed. State whether the phrase in the docs holds and how it should read if it does not.
4. Every public number and statement: spot-check at least 15 numbers and statements in the README, write-up and paper against the logs, including the ones about reviews, costs, null controls, the calibration rows, the attack count and the seed time range. Flag any statement with no source, and any first-person statement that implies work a person did by hand when an agent did it. Flag em dashes, private names or paths, and any claim of being first.
5. A stranger's first minute: does the README make sense in order, does `reproduce.py` do what it says (read it; run `reproduce.py series` which is cheap, and tests only if seconds), and is there anything that would not work from a clean clone (paths, files ignored by .gitignore that the docs need, workflows that assume a private repo)?

LIMITS: The Mac is under load and another job holds its training lock: no `verify.py` at the real budget and no training. Reading, small python, and `git log` only. Edit nothing except your one output file, no commits. Be economical: search, do not read transcripts through.

OUTPUT: write `review/final-review.md`: plain prose, numbered findings ordered by how much each would hurt, each with its evidence and the exact fix you suggest, then a section "What holds up", then a line starting `Could not check:`. Under 80 lines. No em dashes. Every number must be one you read or computed.

Reply in under 250 words: whether it should go public as it stands, and the three findings that matter most.
```
