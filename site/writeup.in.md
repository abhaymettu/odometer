# odometer: a training speedrun with a meter where the clock would be

Training speedruns time you, so a result measures the idea and the GPU together, and a number from one machine means little on another. I wanted a track where hardware can't buy a better score, so that people and coding agents could sit on one record table and anyone could check a record at home. I built it with Claude Code, which wrote the engine, the verifier, the recipes and the harness under my direction.

odometer charges every tensor operation a price in ticks, worked out from the tensor shapes alone. An entry gets 40 billion ticks to train a byte-level model on TinyStories, and its score is held-out bits per byte. A laptop and a server stop at the same reading, with scores that agree far below the record margin, and a faster machine only gets there sooner. [The demo](https://abhaymettu.github.io/odometer/) shows this directly: it runs the current record's code in your tab at a smaller budget and stops on the same tick as native code.

The current record is {record_test} bits per byte, against {recipe_test} for the best recipe ported from the literature, and the {agent_records} newest rows of the table came from Claude Code agents. From a clean clone, `python reproduce.py` reruns it on fresh seeds in {quick_seconds} seconds on a 2-core runner.

## Why a meter

The obvious way to remove hardware is to count FLOPs, and that doesn't work. Calibrated to time, a matmul costs one tick per 128 multiply-adds and `exp` one tick per element, so under a flat FLOP price an entry heavy on elementwise work would take about a hundred times longer for the same reading, and an agent would find that gap fast.

So the prices track time instead. A tick is about one nanosecond of one laptop core, and each price is a function of shapes, so a program that doesn't branch on its floats reaches the same reading anywhere. The same seeds of the current record ran on {fair_machines} machines: ticks matched exactly on all of them, and scores differed by at most {fair_max} bits per byte, about {fair_ratio} times below the record margin. {branch_note}

Prices are close to time and not equal to it. On x86, 331 of 335 calibration rows sit within 5x of the median, a copying `reshape` runs about 7x over its price, and squaring runs about 7x under its own. The prices were frozen once records stood on them, so those misses stay.

## What a record has to survive

A new record has to beat the holder by more than 0.002 bits per byte at p < 0.01 on 8 fresh seeds, do it again on 8 new seeds, and then get a written review that explains what buys the gain. Training runs in a process that never loads held-out bytes, and scoring runs in a second, fresh one. An entry can import only `odometer` and `math`. A quarter of the scored windows have a twin that shares a prefix and differs after a random cut, so if the logits before the cut move, the entry was looking ahead and the run is void.

The repo holds {attacks} entries written to cheat, and a test pins the exact way each one fails. Four still work, and the tests pin those too. Three get compute for free, though the ration on training reads leaves that compute little to learn from, which is inferred and not measured. The fourth is a look-ahead bonus under the twin tolerance, worth under 1.5e-4 bits per byte.

## What the agents did

{attempts} fresh Claude Code sessions ran one after another, each with Opus 5.5 at high effort, 40 minutes on a 4-core VM, and the current record to start from. The plan and the count were committed before the first one. The test split was zeroed in each agent's copy of the data, so a claim first met test data in a rerun in CI, after the session had ended.

{records} of {claims} claims cleared the record rule, with gains falling from {best_gain} to {gain_last} bits per byte. All {survived} replicated on 8 fresh seeds (Wilson 95% interval {claim_lo} to {claim_hi}, and {attempt_survived} of {attempts} attempts, {attempt_lo} to {attempt_hi}), but that count could hardly have come out otherwise. Dev and test are a random split of one file, and in simulation, had every claim been exact, all {claims} would replicate with probability above 0.999, and at half the claimed gain still about {p_half}. So it mainly rules out a broken pipeline or misreported dev means, and says little about whether agent claims hold where failure is possible. Claimed gains didn't visibly shrink either (median verified over claimed {shrink}, {shrink_below} of {claims} below 1), though the plan set no threshold and this size can't resolve it. Opus was chosen after its pilot passed and Sonnet's pilot claim did not.

Every score reads the same 1024 windows, so the plus-minus covers seed noise on those windows only. Rescoring each record from {fresh_first} on with the next 1024 test windows, which no score had read, gives {fresh_fresh_gain} bits per byte of gain to the current record against {fresh_used_gain} on the usual windows.

None of them replaced the model. Hashed context mixing was already the best recipe at this budget, and each record works inside it: fixing a padding artefact at the start of a window, a learned gate in place of hand-set constants, word contexts, fitting that gate on fresh bytes, thinner high-order tables. A lot of this is specific to the track. About a third of record 07's gain (0.0061 of 0.0187, one seed, dev) sits in the first 16 of 512 positions of a window, and about three quarters of the gain in attempt-04's record 09 is eval-time modelling, paid for with 36.5% more eval ticks. The reviews also caught agents being wrong about their own work: one explanation was only half right, and another credited a change that cost ticks, where dropping stages was what paid for the gain.

A session cost ${cost_lo} to ${cost_hi} at list price, and every transcript is in the repo.

## What it doesn't settle

The recipes at the bottom of the table were ported and tuned with a coding agent too, and no person has run this track, so nothing here compares agents with people. Fresh Claude sessions wrote the reviews that decide a record, and none has refused one yet. {attempts} attempts on one model and one prompt say nothing about agents in general.

The record rule is strict at the margin and weak just above it: at a seed SD of 0.001 it passes a true gain of 0.003 about {rule_003} of the time and 0.004 about {rule_004}. Resubmitting each starting record unchanged gave {null_false} false positives in {null_n} runs, which catches a broken pipeline and nothing finer. There is no memory or disk cap. The test split is public, so outside the series a record rests on published logs and a review and not on a seal, and a table of test text in the source could reach a ceiling of 0.0625 bits per byte, which nobody has measured.

The nearest precedent I know of is [ARC's White-Box Estimation Challenge](https://www.lesswrong.com/posts/Kben8CzS4awCwNw5c) (2026), whose public leaderboard ranks estimators on FLOPs counted per operation, so an estimator reads the same count on a laptop and on the grader. It meters estimation and not training. Green AI (2019) proposed operation counts as a hardware-independent measure of work, and the MicroNet Challenge (2019) scored inference that way. I found no public record track that meters the training run itself, from about a dozen searches, which is not proof. Survival of agent claims has been measured before, by "Separating Real Improvements from Lucky Runs in Autonomous ML Research" and, for prompts, "The Winner's Curse in LLM Self-Improvement Loops", and the single `train.py` format comes from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
