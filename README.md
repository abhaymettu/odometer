<img src="docs/fig/ladder.svg" alt="The record table: test bits per byte falling with each new record, down to the entry an agent wrote." width="100%">

# odometer

A speedrun for training a small language model, with a meter where the clock would be.

<!-- quick -->
**From a clean clone to a verified result in ? seconds on a 2-core runner ([log](https://github.com/abhaymettu/odometer)).** The current record scores 1.0887 bits per byte, and the 6 newest rows of the table came from Claude Code agents. Built with Claude Code under my direction, including the engine, the verifier, the recipes and the harness, and fresh Claude sessions wrote the reviews. No person has run this track yet.
<!-- /quick -->

Every tensor operation costs ticks, priced from its shape alone. An entry gets 40 billion ticks to train a byte-level model on TinyStories, and its score is held-out bits per byte. A laptop and a server run the same program to the same reading, and their scores agree within a tolerance far below the record margin, so hardware only buys finishing sooner. A record has to survive a rerun on fresh seeds by a verifier you can read in one sitting, and coding agents sit on the same table as everyone else.

**[Run the record in your browser](https://abhaymettu.com/odometer/)**. It takes about half a minute, and the page shows your device stopping at the same tick as every other.

## Try it

You need Python 3.12 or newer.

```bash
git clone https://github.com/abhaymettu/odometer && cd odometer
python3 -m venv .venv && .venv/bin/pip install numpy==2.5.4
source .venv/bin/activate
python reproduce.py
```

That downloads the data, checks it against fixed hashes, reruns the current record on 2 fresh seeds on the test split, and prints your number beside the stored one. Every other headline number has its own command:

| Number | Command |
|---|---|
| The whole table, 8 fresh seeds per record | `python reproduce.py table` |
| Every adversarial entry gets rejected or priced out | `python reproduce.py attacks` (needs `pytest`) |
| The agent series, from the stored claim logs | `python reproduce.py series` |
| Scores across machines | `python tools/fairness.py study/fairness/logs/*.json` |
| What each op really costs on your machine | `python tools/calibrate.py` |
| Budget sweep | `python tools/sweep.py records/11-mixer-thin/train.py --out sweep.json` |
| How much the series' numbers could have varied | `python tools/power.py` |
| Records on test windows no score had read | `python tools/fresh_windows.py records/11-mixer-thin/train.py --seeds 7001,7002 --out fresh.json` |

## The table

<!-- records -->
|  | Record | Test bits per byte, mean ± seed SD on 8 seeds | From |
|---|---|---|---|
| 11 | [Mixing, thinner high-order tables, cheaper gate](records/11-mixer-thin) | 1.0887 ± 0.0008 | Claude Opus 5.5, attempt-06 of the series |
| 10 | [Mixing, gate fitted on fresh bytes](records/10-mixer-fresh) | 1.0930 ± 0.0010 | Claude Opus 5.5, attempt-05 of the series |
| 9 | [Mixing, words and story boundaries](records/09-mixer-words) | 1.0976 ± 0.0014 | Claude Opus 5.5, attempt-04 of the series |
| 8 | [Mixing, learned in-window gate](records/08-mixer-gate) | 1.1065 ± 0.0007 | Claude Opus 5.5, attempt-02 of the series |
| 7 | [Mixing, cold-start fix](records/07-mixer-cold) | 1.1256 ± 0.0007 | Claude Opus 5.5, attempt-01 of the series |
| 6 | [Mixing, folded head, lazy tables](records/06-mixer-fold) | 1.1440 ± 0.0012 | Claude Opus 5.5, in a pilot before the series |
| 5 | [Hashed context mixing](records/05-mixer) | 1.1555 ± 0.0010 | Recipe after Mahoney's PAQ and Knoll's cmix, ported and tuned here |
| 4 | [PPM with learned backoff](records/04-ppm) | 1.2688 ± 0.0003 | Recipe after Cleary and Witten's PPM, ported and tuned here |
| 3 | [Modern transformer](records/03-modern) | 1.5027 ± 0.0070 | Recipe from recent small-model practice, ported and tuned here |
| 2 | [Character RNN](records/02-rnn) | 1.5442 ± 0.0130 | Recipe after Karpathy's char-rnn, ported and tuned here |
| 1 | [Small GPT](records/01-gpt) | 1.7058 ± 0.0106 | Recipe after Karpathy's nanoGPT, ported and tuned here |
<!-- /records -->

Test bits per byte on 8 fresh seeds that played no part in seating that record. Later claims are compared with them. The ± is seed noise on these 1024 windows only. The first five are recipes: each one ported from its authors and tuned on the dev split with a coding agent, and the log of all that tuning is in [entries/TUNING.md](entries/TUNING.md). No person ran them on this track, so read them as baselines and not as human records. Everything after them came from an agent. The sixth came from a pilot before the registered series, and the others from the series itself.

A new record has to beat the holder by more than 0.002 bits per byte at p < 0.01 on 8 fresh seeds, do it again on 8 new seeds, and then get a written review that explains why the gain is real. Each `records/<name>/` holds the code, every seed's JSON log, and that review.

## Why a meter

Speedruns time you, so they measure the hardware along with the idea, and a result from one machine means little on another. Pricing by FLOPs doesn't fix it for training. Calibrated to time, a matmul costs one tick per 128 multiply-adds and `exp` one tick per element, so under a flat FLOP price an entry heavy on elementwise work would take about a hundred times longer for the same reading, and an agent would find that gap fast.

So the price list in [`odometer/_engine.py`](odometer/_engine.py) tracks time instead. A tick is about one nanosecond of one laptop core, and each price is a function of tensor shapes, so a program that does not branch on its floats reaches the same reading anywhere. [`tools/calibrate.py`](tools/calibrate.py) measures every op against its price and reports any that drift more than 5x from the median. One seed of a record takes about 70 to 155 seconds on a 2-core CI runner, and 43 to 174 seconds on the 4-core VM. The rows are in [study/calibration/x86.json](study/calibration/x86.json).

## What a rerun guarantees

[`verify.py`](verify.py) is the referee, and it's short enough to read. Its docstring lists each guarantee.

- Training runs in its own process that never loads held-out bytes. Scoring runs in a second, fresh process.
- An entry can import only `odometer` and `math`, and it can name only the engine's public API.
- Python work outside the engine is counted, so it can't stand in for metered compute. Literals and syntax nodes are rationed, so a table can't ride in the source.
- `forward()` sees held-out inputs and never targets, gets fresh parameters on every call, and has its compute per byte capped.
- A quarter of the scored windows have a twin that shares a prefix and differs after a random cut. If the logits before the cut move, the entry was looking ahead or across rows, and the run is void.

## Beat the record

Write one Python file with `train(data)` and `forward(params, x)`. [`entries/`](entries) has five worked examples, and [SUBMITTING.md](SUBMITTING.md) has the steps.

```bash
python verify.py entries/mine/train.py --smoke --split dev     # seconds, to check that it runs
python verify.py entries/mine/train.py --split dev --seeds 8   # tune on this
python verify.py entries/mine/train.py --against records/current.json
```

## Agents on the same table

<!-- agents -->
6 fresh Claude Code sessions ran one after another, each with Opus 5.5 at high effort, 40 minutes on a 4-core VM, and the current record to start from. The test split was zeroed in each agent's copy of the data, so a claim first met test data in the rerun, in a separate CI job, after the session had ended.

5 of 7 claims cleared the record rule, with gains falling from 0.0188 to 0.0041 bits per byte. All 7 replicated on 8 fresh seeds (Wilson 95% interval 0.65 to 1.00, and 6 of 6 attempts, 0.61 to 1.00), but that count could hardly have come out otherwise. Dev and test are a random split of one file, and in simulation, had every claim been exact, all 7 would replicate with probability above 0.999, and at half the claimed gain still about 0.69. So it mainly rules out a broken pipeline or misreported dev means, and says little about whether agent claims hold where failure is possible.

<img src="docs/fig/claims.svg" alt="Claimed gain against verified gain for each agent claim, with 95 percent intervals. Most verified gains sit on the claimed gains." width="100%">

| Attempt | Started from | Claimed gain | Verified gain ± 95% half-width | Outcome | Record |
|---|---|---|---|---|---|
| 01 | 06-mixer-fold | 0.0184 | 0.0185 ± 0.0011 | replicated | yes |
| 02 | 07-mixer-cold | 0.0172 | 0.0188 ± 0.0006 | replicated | yes |
| 03 | 08-mixer-gate | 0.0026 | 0.0025 ± 0.0008 | replicated | no |
| 04 | 08-mixer-gate | 0.0091 | 0.0095 ± 0.0008 | replicated | yes |
| 05 | 09-mixer-words | 0.0055 | 0.0047 ± 0.0013 | replicated | yes |
| 06 | 10-mixer-fresh | 0.0037 | 0.0034 ± 0.0008 | replicated | no |
| 06 | 10-mixer-fresh | 0.0041 | 0.0041 ± 0.0009 | replicated | yes |

Claimed gains did not visibly shrink: median verified over claimed 0.996, 4 of 7 below 1, range 0.86 to 1.09. The plan set no threshold, and shrinkage cannot be resolved at this size. Opus was chosen for the series after its pilot passed and Sonnet's pilot claim did not.

Every score reads the same 1024 windows, so every plus-minus here covers seed noise on those windows only. As a check, each record from 05 on was rescored on the next 1024 test windows, which no score had read, on 8 new seeds. Gains from record 05 to the current one: 0.0665 on the usual windows and 0.0663 on the new ones. For the last record alone: 0.0041 and 0.0043.

| Record | Usual 1024 windows | Next 1024 windows | Difference |
|---|---|---|---|
| Hashed context mixing | 1.1555 | 1.1550 | -0.0004 |
| Mixing, folded head, lazy tables | 1.1444 | 1.1442 | -0.0003 |
| Mixing, cold-start fix | 1.1255 | 1.1259 | +0.0004 |
| Mixing, learned in-window gate | 1.1069 | 1.1062 | -0.0007 |
| Mixing, words and story boundaries | 1.0976 | 1.0974 | -0.0001 |
| Mixing, gate fitted on fresh bytes | 1.0931 | 1.0930 | +0.0000 |
| Mixing, thinner high-order tables, cheaper gate | 1.0890 | 1.0887 | -0.0002 |

Each record also needs a written review from a fresh Claude session, which reads the code and logs and has to say what buys the gain. Much of the gain is tied to this track: about a third of record 07's gain (0.0061 of 0.0187, one seed, dev) sits in the first 16 of 512 positions of a window, and about three quarters of the gain in attempt-04's record 09 is eval-time modelling, paid for with 36.5% more eval ticks. A review can refuse a record, but none has: all six on agent records returned real, even the one that found the agent's account of its own change wrong. The briefs are in [study/reviews.md](study/reviews.md). What each one wrote is in `records/<name>/review.md`.

As a check on the pipeline, each starting record went through it unchanged, 8 times over: 0 false positives in 40 runs across 5 records, which catches a broken pipeline and nothing finer. 1 run came back refuted, on record 08, whose stored batch of 8 seeds sits about 2.1 standard errors below the mean of its nulls, so gains measured against that record read about 0.0005 low. The sessions cost $2.21 to $4.49 each at list price.
<!-- /agents -->

The plan, the count and the classification rule were committed before the first counted attempt: [study/PREREG.md](study/PREREG.md). Each agent's whole session is in `study/attempts/<id>/`.

## Same program, same reading

<!-- fair -->
For a program that does not branch on its floats, ticks are a function of the program, so they have to match exactly. Scores pass through each machine's BLAS, so they can differ in the last digits. Same seeds of `?`, each machine against ?, with a tolerance of ? fixed before the machines ran:

| Machine | Same seeds | Ticks identical | Largest difference, bits per byte | Paired TOST p |
|---|---|---|---|---|

The largest difference is ? bits per byte, about ? times below the record margin of 0.002. ? Rerun the comparison with `python tools/fairness.py study/fairness/logs/*.json`, or run the matrix yourself with [the fairness workflow](.github/workflows/fairness.yml) and download its artifacts. In a browser the demo reads the same tick as native code.
<!-- /fair -->

## What each tick buys

<img src="docs/fig/sweep.svg" alt="Dev bits per byte against training budget for seven of the records. Hashed mixing stays lowest at every budget." width="100%">

Recipes keep the settings they were tuned with at 40 billion ticks, so this is a floor on what each could do at another budget. Hashed context mixing stays lowest at every budget run. The small GPT and the modern transformer keep falling, and the modern transformer passes PPM somewhere past the track's budget. PPM gets worse at 160 billion. It has read the 64 MiB file about six times by then, and its counts may grow overconfident, which was not tested.

## How a cheat dies

[`tests/attacks/`](tests/attacks) holds 79 entries written to cheat: reading held-out bytes, looking ahead, keeping state between calls, getting compute for free, shipping data in the source, and breaking the verifier. [`tests/test_attacks.py`](tests/test_attacks.py) pins the exact way each one fails, and [the attacks README](tests/attacks/README.md) names each hole the suite found, with its fix.

Four still work, and the tests pin those too, so a change in either direction shows up:

- `open_c_level_sum`: `sum(range(n))` is one counted line, so only the wall limit stops it.
- `open_bigint_vector`: Python big integers do vector arithmetic with almost no ticks.
- `open_long_line`: a line with 2000 additions counts once.
- `open_below_tolerance`: a look-ahead bonus just under the twin tolerance passes, for under 1.5e-4 bits per byte, 13 times below the record margin.

The first three are free compute, but `item()` rations training bytes to 20,000 reads, so free compute has almost nothing to learn from. That this removes any score gain is my inference. I did not measure it.

## What it does not do

- It has no memory or disk cap. The PPM recipe peaks near 1.2 GB.
- Twin values reached 2.1e-05 across 656 stored seed runs at the track's budget, against a tolerance of 1e-4. They look like host float noise, since the same code often gives exactly 0 on another host, but the cause was not found.
- The test split is public. `prepare.py` downloads it, the permutation seed is in the code, and `--against` gives a test score as often as you like, so a record rests on published logs and a review and not on a seal. In the agent series the split was zeroed in each agent's copy. The obvious attack is a table of test text in the source: the 4096-byte literal cap allows 32,768 bits against 524,288 scored bytes, a ceiling of 0.0625 bits per byte, 31 times the margin. How much of that a real table gets is unmeasured, and it would sit in plain view of a review. During tuning, two smoke runs read 32 test windows by accident, and [the tuning log](entries/TUNING.md) records them.
- Prices are close to time, not equal to it. On x86, 331 of 335 calibration rows sit within 5x of the median. A copying `reshape` and its backward run about 7x over their price, so an entry can spend more wall time than its ticks suggest, though copying teaches a model nothing. Squaring and `embedding` run about 6x to 7x under theirs. I froze the prices once records stood on them, so `tools/calibrate.py` exits nonzero on x86.
- The score is one corpus and one budget. A 100-byte alphabet that helps here would cost dearly on anything else.
- At a seed SD of 0.001 the record rule passes a true gain of 0.003 about 14% of the time and 0.004 about 82%, so a gain of 1.5 times the margin usually fails. The weaker `replicated` test passes one seed SD about 29% of the time. Both come from `python tools/power.py`.

## Prior work

The single `train.py` and the bits-per-byte framing come from [karpathy/autoresearch](https://github.com/karpathy/autoresearch), whose README notes its wall-clock results aren't comparable across compute platforms. [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) and OpenAI's Parameter Golf set the pattern of statistical record rules, and Prime Intellect's verifier reruns 8 seeds. Survival of agent claims has been measured before too, by "Separating Real Improvements from Lucky Runs in Autonomous ML Research" and, for prompts, "The Winner's Curse in LLM Self-Improvement Loops". The nearest precedent I know of is [ARC's White-Box Estimation Challenge](https://www.lesswrong.com/posts/Kben8CzS4awCwNw5c) (2026), whose public leaderboard ranks estimators on FLOPs counted per operation by a metered NumPy wrapper, so an estimator reads the same count on a laptop and on the grader. It meters estimation and not training. Green AI (2019) proposed operation counts as a hardware-independent measure of work, and the MicroNet Challenge (2019) scored inference that way. I found no public record track that meters the training run itself, from about a dozen searches, which is not proof. Those searches are in [review/novelty-and-exploits.md](review/novelty-and-exploits.md).

Recipes credit their authors in each file: nanoGPT (Karpathy), char-rnn (Karpathy) and GRU (Cho et al.), modded-nanogpt (Jordan et al.), PPM (Cleary and Witten; Moffat), PAQ and lpaq (Mahoney), cmix (Knoll). Data: TinyStories (Eldan and Li, 2023), CDLA-Sharing-1.0.

## Layout

```
verify.py          the referee
odometer/          the metered engine and data access
entries/           the recipes as tuned, and the tuning log
records/           every record: code, seed logs, review, verdict
tests/attacks/     entries written to cheat
study/             the agent series: plan, harness, every attempt's session and claims
tools/             calibration, budget sweep, fairness, figures, page build
docs/              the page, and the in-browser demo
```

MIT licence. The data keeps its own.
