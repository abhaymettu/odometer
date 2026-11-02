# gpt: 1.7171 dev bpb

1. Mechanism. Starvation sets the number. At 19,883 ticks per byte (measured at the smoke budget, then scaled) it draws 2.0 M of the 66.0 M training bytes in 2,620 updates. Matmuls and their backward (lines 60-62, 70-71) take 35% of ticks. The 96-float token row costs about 1,540 ticks per byte at 16 per float (`_engine.py:806`). Eval sees at most 48 bytes of context (lines 81-85).

2. Ladder. First row, nothing to beat. It is a recipe result, not a transformer limit: `modern` has the same width, layers, heads and block and scores 0.209 lower.

3. Abuse. None found. `d * d` and the added mask (lines 50, 61, 75) compute the same maths at listed prices. Twin difference is 0.0. Eval costs 16,255 of 50,000 ticks per byte, at 0.75 ns per tick on an M4 against a calibration median of 0.60.

Could not check: how it ranks at higher budgets. TUNING has no budget sweep for the neural rows.

Verdict: real
