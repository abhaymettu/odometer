# ppm: 1.2692 dev bpb

1. Mechanism. Lines 59-61 sum table lookups and call `backward()`, so the scatter-add leaves counts in `.grad`. At 414 ticks per byte it draws about 97 M bytes, 46 times modern. Line 77 folds them into 3.82 M floats. Eval does 1,020 lookups per byte at 8.25 ticks (line 112).

2. Against modern: 0.239 better, because stored phrases beat learning at this budget. 86.9% of dev order-12 contexts occur in train, only 1.9% of order-64, so phrases, not duplicated stories. Votes, absent from modern, gave 0.052 at smoke.

3. Abuse. A rule gap, not a mispricing. Wide tables hold 66.7 M floats, 16 times the cap, since `verify.py:262` checks only returned floats. They cost 31% of ticks. On an M4, scatter runs 2.20 ns per tick at 33.5 M floats, forward 0.09, together 0.50 (median 0.60). Eval matches the logged 24,082 ticks, at 0.43 ns per tick. Only the verifier's normalization escapes the cap, and it changes nothing.

Could not check: the score with wide tables capped at 4.19 M floats.

Verdict: real
