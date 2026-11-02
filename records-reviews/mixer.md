# mixer: 1.1566 dev bpb

1. Mechanism. Nine hashed orders up to 12 (lines 54-58) each fetch an 8-float row, about 138 ticks per lookup, and embedding backward takes 34% of train ticks. It draws 15.5 M bytes, 11.3 M trained, and one hidden layer mixes them (lines 64-65).

2. Against ppm: 0.113 better. Not data (ppm draws six times more), not floats (4.10 M against 3.82 M), not eval (same votes formula, same 9,941 ticks per byte), not price (embedding backward 0.36 ns per tick, median 0.60). Ppm leads at 2e9 ticks (1.5741 against 1.6015, same windows), so training creates the gain. That learned rows beat fixed counts is my inference.

3. Abuse. None found. `window_votes` takes 72% of eval ticks, its matmul matches the hand price, and it costs 0.41 ns per tick. Including the diagonal voids on the twin check, and 24 stages void at the cap. Votes cost no train ticks and gave 0.073 at smoke.

Could not check: whether ppm tuned as hard closes the gap (6 sweep rows against 65).

Verdict: real
