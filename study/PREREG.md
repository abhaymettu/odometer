# odometer agent series: preregistration

Status: registered 2026-11-02, before the first counted attempt. The first commit that holds this file with
`PLANNED_ATTEMPTS` and `MODEL` filled in is the registered version. Any later change is listed under Deviations, with its
date and reason.

PLANNED_ATTEMPTS: 6
MODEL: claude-opus-5-5, effort high
WALL_CLOCK_MINUTES: 40

## Question

When Claude Code agents improve a training recipe on this track and claim a gain, how often does the claim survive an
independent rerun on fresh seeds on a sealed split, and how much of the claimed gain is left?

## Hypotheses

1. The survival rate is below 1. This is an estimate with an interval, not a test against a threshold. No target value
   is set in advance.
2. Claimed gains shrink. The median of verified gain over claimed gain is below 1.
3. The pipeline's false-positive rate on unchanged entries is compatible with the nominal 0.01. The null control can
   detect only a gross failure of this, as explained below.

## Design

An attempt gives a single agent a model, a wall-clock allowance, a fresh directory and a starting record. The agent gets the
prompt in `study/prompt.md`, and the following:

- A directory with the track's code, the entries, and the training data. `sandbox.COPIED` is the exact list. It does not
  hold this file, the design notes, the review of exploits, earlier attempts, or any test-split number.
- A copy of `heldout.bin` in which every byte outside a dev window is zero. Dev windows are unchanged.
- An empty HOME, and an environment with only PATH, HOME, LANG, CLAUDE_CODE_OAUTH_TOKEN and
  CLAUDE_CODE_SUBPROCESS_ENV_SCRUB, plus the PWD that bubblewrap sets. There is no user config, hook, memory or CLAUDE.md.
  The skills and plugins built into the Claude Code binary remain, and `init` in `attempt.json` lists them.
- Permission to use every tool without asking, through `--allowedTools`. With the scrub variable set, Claude Code 2.1.316
  overrides `bypassPermissions` to `default`, so the list is what grants the tools.
- A filesystem view (bubblewrap) with the system's programs, its own directory, the Python environment read-only, and
  nothing else from the machine. It has its own process namespace. It keeps network access, including the web.

Before any model call, a script inside that view checks the environment, HOME, what is visible, and the zeroed split,
and stops the attempt if any check fails. Its report is kept as `isolation.json`. Then `claude -p` runs once with an
isolation prompt in a throwaway HOME, and its output is kept as `probe.jsonl`. The attempt stops there if the probe does not
finish, if its Bash can see the token, or if any tool was denied. Denied tools during an attempt are reported per attempt.

Attempts run one at a time on one 4-core VM (AMD Ryzen 7 9700X). Nothing else runs on that machine during an attempt. A
record seated by an earlier attempt is the starting record for later attempts, so each attempt's claims are verified, and
any record seated, before the next attempt starts.

The agent may write up to 3 claims. After the agent has exited, each claim is copied out and run through the repository's
own `verify.py`, on the test split, on 8 fresh seeds drawn by the verifier, with `--against` the attempt's starting record.
Verification runs in GitHub Actions on a 2-core ubuntu-24.04 runner, not on the attempt's VM, from the same code hash.

## Classification

Let c be the claimed gain, which is the starting record's dev mean minus `claimed_dev_bpb`. Let g be the verified gain, which is
the mean of the starting record's 8 stored test seeds minus the mean of the claim's 8 test seeds. The test is Welch's, as in
`stats.welch`. `rules.classify` applies these outcomes in this order:

1. `rejected`: the verifier voided the claim, meaning a static failure, a void seed, or a verifier crash. A claim with a
   missing or malformed `claim.json` is also `rejected`.
2. `replicated`: the one-sided p that g > 0 is below 0.01.
3. `refuted`: not replicated, and the upper bound of the two-sided 95% interval of g is below c. The rerun rules out what
   was claimed.
4. `inconclusive`: everything else.

Separately, `record` is true if the claim passes the two-stage rule in `verify.py --against`. Stage 1 tests at p < 0.01 beyond a
margin of 0.002 bpb, and stage 2 repeats that test on 8 new seeds. The classification uses only the first 8 seeds. The ladder
acts on `record`. `replicated` has no margin, so a claim can be `replicated` without being a record.

An outcome, once stored, is final. Rerunning `verify_claims` does only claims with no stored outcome. If the verifier
dies from a signal, no outcome is stored and that step is rerun; this is recorded under Deviations.

## Ladder

A new record must pass the record rule and then a written review by a fresh reviewer, run by hand. The same gate applies to
human recipes and agent entries. After seating, the record is run on 8 fresh test seeds that decide nothing else. Those are
the stored seeds later claims are compared against. Stage 2 seeds of the winning claim are not reused, because they were
selected for passing. The same run also takes 8 dev seeds, and those give the agent the numbers in its prompt. If several
claims in one attempt pass the record rule, only the one with the lowest test mean goes on to review.

## Pilots

Two pilot attempts ran before this file was registered, and their records stay in `study/attempts/`. They are in no
survival figure: `study.analyze` lists folders named `pilot-*` apart and never counts them.

- `pilot-01`: Sonnet 5.5, default effort, 60 minutes, with an earlier prompt whose last line was "Stop when you have nothing
  better to try." It stopped itself at 28 minutes with one claim, verified `inconclusive`.
- `pilot-02`: Opus 5.5, high effort, 40 minutes, with the prompt as it stands now. It used 37 minutes and wrote two claims,
  both of which passed the record rule.

What the pilots changed, all before any counted attempt: the model and effort, the allowance, and the prompt's last line,
now "The time is yours to use. There is no reward for finishing early." The planned count comes from the VM hours left
after the pilots, at about 0.7 hours per attempt, with enough spare for one full replacement and a reserve of 0.5 hours.

The record table is open to anyone, so a pilot claim that passes the record rule and the review is seated like any other,
and it is then the starting record for the series. Where such a record is shown, it is labelled as coming from a pilot.
`pilot-02` claim 2 went through that gate and is seated as `records/06-mixer-fold`, so the series starts from it.

## Outcomes

Primary:

- Survival per claim: replicated divided by all claims, with a Wilson 95% interval.
- Survival per attempt: counted attempts with at least one replicated claim, divided by counted attempts, with a Wilson 95%
  interval. Claims within an attempt are not independent, so this is reported beside the per-claim rate with equal standing.
  Counted attempts with no claim stay in this denominator.

Secondary:

- Median shrinkage g/c over non-rejected claims with c > 0. A ratio over c <= 0 means nothing, so those claims are counted apart.
- Number of records.
- Seed-runs the agent spent before each claim, counted from the run logs in its directory. This is a lower bound, since an
  agent can train some other way.
- Survival broken down by starting record, since a later record is a different problem.
- The largest verified gain.

Null control. Each record that a counted attempt started from is resubmitted unchanged 8 times, 8 seeds each, through the same
scoring with c = 0, using stage 1 only.

## Stopping rule

Run attempts until `PLANNED_ATTEMPTS` have been counted, then stop. The count is not changed after seeing outcomes. Pilot
attempts are not counted, and they end when this file is registered. If the VM hours run out first, the series stops at
the count reached, and that is listed under Deviations.

## Exclusions

An attempt that dies from infrastructure before writing any claim is logged and not counted. Infrastructure means the agent
process exited with an error or with no final result event, or the isolation gate or probe failed. It is replaced by a new
attempt. Everything else counts, including an attempt that ran out of time, an attempt that wrote no claim, and an attempt
that died after writing a claim. A claim that crashes the verifier counts as rejected.

## If agents fail to beat the baseline

Agents have failed to beat the baseline if none of the claims is `replicated` and none sets a record. That result is reported
the same way as any other: the counts, the survival rate with its interval (with 40 claims and none replicated, the Wilson upper
bound is 0.088), and the largest verified gain with its interval. It is not softened, and the series is not extended to look for
a different answer.

## Known limits

- **Power.** 8 against 8 at one-sided 0.01 passes a true gain of 1 seed SD 31% of the time, 1.5 SD 65%, and 2 SD 90%
  (simulation in `review/novelty-and-exploits.md`, item 4). `inconclusive` often means too small to resolve, not false.
- **Shared comparison batch.** Every claim against a record is compared with the same 8 stored seeds, so an unlucky stored
  batch moves all of those claims together. The per-claim Wilson interval does not account for this. The null control
  reports how far the stored batch sits from the mean of the record's null seeds, in standard errors, so a reader can see it.
- **Null control resolution.** 8 nulls cannot confirm a rate of 0.01. A result of 0 of 8 has a Wilson upper bound of 0.324. It
  catches a broken pipeline and nothing finer. Comparing the null batches with each other gives 56 comparisons, which are
  not independent, and that figure is reported as a rough check only.
- **No multiplicity correction.** The classification is per claim. Among many null claims, about 1 in 100 would be
  `replicated` by chance.
- **Seal.** The seal prevents accidental use of the test split. TinyStories is public and the agent has the web, so an agent
  could rebuild the split on purpose. The transcripts are published, so this could be seen, though it could not be undone.
  Static rules in `verify.py` limit what such knowledge could buy (review items 7 and 15).
- **Network.** The agent shares the machine's network, so services on localhost are reachable from inside its view.
- **Repository.** The repository stays private until the series ends, or an agent could find this file on the web.
- **Compute.** The training budget is machine independent, but the agent's allowance is wall clock on one 4-core machine, so
  the number of experiments an agent can run depends on that machine.
- **Generality.** The result applies to the registered model, prompt and allowance on this track, not to agents in general.

## What is not claimed

The survival fraction of agent claims has been measured before. This series is a further measurement on an open record table,
not a first.

- "Separating Real Improvements from Lucky Runs in Autonomous ML Research" (NeurIPS 2026 Workshop for Autonomous Machine
  Learning Research, poster) reports that of 15 non-null candidates in a nanoGPT setup, 8 replicated on sealed seeds, 3 failed and
  4 stayed inconclusive. It also found that gains near the noise scale stayed unresolved after 24 to 33 sealed runs.
- "Winner's Curse in LLM Self-Improvement Loops" (arXiv 2610.09239) measured the winner's curse for keep-if-better loops on
  prompts, not on training. Scoring start and current on 64 never-used items removed the average bias.

This summary rests on `review/novelty-and-exploits.md`, which read the abstracts of these two works. Survival here also does
not show that a gain has a real mechanism behind it, that it holds at other budgets, or that an agent understood why it worked.

## Deviations

No deviations have been recorded.
