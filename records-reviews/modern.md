# modern: 1.5080 dev bpb

1. Mechanism. Same width, layers, heads and block as gpt at nearly the same 19,106 ticks per byte, so model size is not the difference. Batch 32 leaves half gpt's updates (1,363), yet it ends 0.209 lower, so per-update progress drives it. Muon's orthogonalized steps (lines 97-100, 126-127) take 16% of ticks. TUNING reports that dropping QK-norm (line 65) cost 0.32 at an earlier config.

2. Against rnn: 0.033 better with 0.55 times the bytes and 0.18 times the updates. It also leads at 2e9 ticks (2.97 against 4.42). The margin is thin: both CIs cover seeds on identical bytes, and the VM log gave modern 2.66 times rnn's wall seconds for equal ticks, though an M4 shows parity (0.66 and 0.64 ns per tick).

3. Abuse. None found. Orthogonalization pays normal matmul prices. Eval costs 14,578 of 50,000 ticks per byte, at 0.72 ns per tick.

Could not check: whether 0.033 survives other bytes. No full-budget parameters exist for a per-window comparison, and I did not read test.

Verdict: real
