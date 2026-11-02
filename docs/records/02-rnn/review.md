# rnn: 1.5407 dev bpb

1. Mechanism. Cell lines 32-36 run once per byte, so the 2,000-tick call price is its largest cost: 36% of train ticks (gpt 9.8%) and 55% at eval. An update every 16 bytes (line 63) yields about 7,400 updates and 3.79 M bytes at 10,554 ticks per byte. `forward` carries state across all 512 bytes (line 51).

2. Against gpt: 0.176 better, with 2.8 times the updates, 1.9 times the bytes and a full window of context instead of 48. That is real, but it is not recurrence beating attention, since `modern` beats both at gpt's exact size. At 2e9 ticks gpt leads (3.57 against 4.42, same windows), so the lead forms late.

3. Abuse. None found, and no pricing gift. The VM log gave it 44 s against gpt's 112 s for equal ticks. An M4 shows near parity (0.64 and 0.56 ns per tick).

Could not check: how the gain splits between updates and context. That needs full-budget parameters rerun with state reset every 48 bytes.

Verdict: real
