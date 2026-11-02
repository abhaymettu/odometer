# odometer: a training speedrun with a meter where the clock would be

Training speedruns time you, so a result measures the idea and the GPU together, and a number from one machine means little on another. I wanted a track where hardware can't buy a better score, so that people and coding agents could sit on one record table and anyone could check a record at home.

odometer charges every tensor operation a price in ticks, worked out from the tensor shapes alone. An entry gets 40 billion ticks to train a byte-level model on TinyStories, and its score is held-out bits per byte. A laptop and a server stop at the same reading with the same score, and a faster machine only gets there sooner. [The demo](https://abhaymettu.com/odometer/) shows this directly: it runs the current record's code in your tab at a smaller budget and stops on the same tick as native code.

The current record is {record_test} bits per byte, and the {agent_records} newest rows of the table came from Claude Code agents. From a clean clone, `python reproduce.py` reruns it on fresh seeds in {quick_seconds} seconds on a 2-core runner.

## Why a meter

The obvious way to remove hardware is to count FLOPs, and that doesn't work. BLAS runs about a hundred times more FLOPs per second than `exp`, so an entry heavy on elementwise work would take a hundred times longer for the same reading, and an agent would find that gap fast.

So the prices track time instead. A tick is about one nanosecond of one laptop core, and it's a pure function of shapes, so the same program reaches the same reading anywhere. I checked that with the same seeds of the current record on {fair_machines} machines: ticks matched exactly on all of them, and scores differed by at most {fair_max} bits per byte, about {fair_ratio} times below the record margin.

Prices are close to time and not equal to it. On x86, 331 of 335 calibration rows sit within 5x of the median, a copying `reshape` runs about 7x over its price, and squaring runs about 7x under its own. I froze the prices once records stood on them, so those misses stay, and the repo's calibration check reads as failed on x86.

## What a record has to survive

A new record has to beat the holder by more than 0.002 bits per byte at p < 0.01 on 8 fresh seeds, do it again on 8 new seeds, and then get a written review that explains what buys the gain. Training runs in a process that never loads held-out bytes, and scoring runs in a second, fresh one. An entry can import only `odometer` and `math`. A quarter of the scored windows have a twin that shares a prefix and differs after a random cut, so if the logits before the cut move, the entry was looking ahead and the run is void.

The repo holds {attacks} entries written to cheat, and a test pins the exact way each one fails. Four still work, and the tests pin those too. Three get compute for free, though the ration on training reads leaves that compute little to learn from, which is my inference and which I did not measure. The fourth is a look-ahead bonus under the twin tolerance, worth under 1.5e-4 bits per byte.

## What the agents did

{attempts} fresh Claude Code sessions ran one after another, each with Opus 5.5 at high effort, 40 minutes on a 4-core VM, and the current record to start from. I committed the plan and the count before the first one. The test split was zeroed in each agent's copy of the data, so a claim first met test data in a rerun in CI, after the session had ended.

{survived} of {claims} claims held up on 8 fresh seeds (Wilson 95% interval {claim_lo} to {claim_hi}), and {records} became records. The plan predicted that claimed gains would shrink, and they did not: the median verified gain was {shrink} times the claimed one. These gains sit far above seed noise, where selection has little room to inflate anything, and I wouldn't expect that to hold nearer the noise.

None of them replaced the model. Hashed context mixing was already the best recipe at this budget, and each record works inside it: fixing a padding artefact at the start of a window, a learned gate in place of hand-set constants, word contexts, fitting that gate on fresh bytes, thinner high-order tables. A lot of this is specific to the track. About a third of one record's gain sits in the first 16 of 512 positions of a window, and about three quarters of another's comes from spending 36.5% more ticks at eval. The reviews also caught agents being wrong about their own work: one explanation was only half right, and another credited a change that cost ticks, where dropping stages was what paid for the gain.

A session cost ${cost_lo} to ${cost_hi} at list price, and every transcript is in the repo.

## What it doesn't settle

The recipes at the bottom of the table were ported and tuned with a coding agent too, and no person has run this track, so nothing here compares agents with people. The reviews that decide a record are written by fresh Claude sessions and not by humans. {attempts} attempts on one model and one prompt give a wide interval and no general statement about agents.

The record rule has limited power: 8 seeds against 8 passes a true gain of one seed SD only 31% of the time, so claims near the noise tend to come back inconclusive. Resubmitting each starting record unchanged gave {null_false} false positives in {null_n} runs, which catches a broken pipeline and nothing finer. There is no memory or disk cap, and the seal on the test split stops accidents and not a determined entrant, since TinyStories is public.

What I couldn't find anywhere else is a training budget metered per operation and independent of hardware, so that's the part to judge this on. Survival of agent claims has been measured before, by "Separating Real Improvements from Lucky Runs in Autonomous ML Research" and, for prompts, "The Winner's Curse in LLM Self-Improvement Loops", and the single `train.py` format comes from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
