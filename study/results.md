# odometer agent series: results

| Measure | Value |
|---|---|
| Attempts counted | 5 of 5 started |
| Attempts excluded (infrastructure, no claim) | 0 |
| Attempts waiting for verify_claims | 0 |
| Counted attempts with no claim | 0 |
| Counted attempts with a denied tool | 0 |
| Claims | 5: 5 replicated, 0 refuted, 0 inconclusive, 0 rejected |
| Survival per claim (primary) | 5/5 (Wilson 95% 0.566 to 1.000) |
| Survival per attempt (primary) | 5/5 (Wilson 95% 0.566 to 1.000) |
| Median shrinkage g/c, claims with c > 0 | +1.005 (n=5, 0 claimed no gain) |
| Records | 4 |
| Seed-runs before a claim, median (range) | 16 (8 to 44) |
| Best verified gain, bpb | +0.0188 |
| Survival from record 06-mixer-fold | 1/1 (Wilson 95% 0.207 to 1.000) |
| Survival from record 07-mixer-cold | 1/1 (Wilson 95% 0.207 to 1.000) |
| Survival from record 08-mixer-gate | 2/2 (Wilson 95% 0.342 to 1.000) |
| Survival from record 09-mixer-words | 1/1 (Wilson 95% 0.207 to 1.000) |
| Null control on 06-mixer-fold | 8 nulls: false replicated 0/8 (Wilson 95% 0.000 to 0.324); false refuted 0; stage 1 passes 0; batch against batch 0/56 (Wilson 95% 0.000 to 0.064); stored batch offset +0.75 SE |
| Null control on 07-mixer-cold | 8 nulls: false replicated 0/8 (Wilson 95% 0.000 to 0.324); false refuted 0; stage 1 passes 0; batch against batch 0/56 (Wilson 95% 0.000 to 0.064); stored batch offset -0.03 SE |
| Null control on 08-mixer-gate | 8 nulls: false replicated 0/8 (Wilson 95% 0.000 to 0.324); false refuted 1; stage 1 passes 0; batch against batch 0/56 (Wilson 95% 0.000 to 0.064); stored batch offset -2.08 SE |

| Attempt | Claim | Base | Claimed gain | Verified gain | 95% half-width | p | Outcome | Record | Seed-runs before |
|---|---|---|---|---|---|---|---|---|---|
| attempt-01 | 1 | 06-mixer-fold | +0.0184 | +0.0185 | +0.0011 | +0.000 | replicated | True | 16 |
| attempt-02 | 1 | 07-mixer-cold | +0.0172 | +0.0188 | +0.0006 | +0.000 | replicated | True | 8 |
| attempt-03 | 1 | 08-mixer-gate | +0.0026 | +0.0025 | +0.0008 | +0.000 | replicated | False | 13 |
| attempt-04 | 1 | 08-mixer-gate | +0.0091 | +0.0095 | +0.0008 | +0.000 | replicated | True | 44 |
| attempt-05 | 1 | 09-mixer-words | +0.0055 | +0.0047 | +0.0013 | +0.000 | replicated | True | 16 |

Pilots ran before the series was registered and count toward nothing above.

- pilot-01 claim 1: 05-mixer, claimed +0.0009, verified +0.0004, inconclusive, record False
- pilot-02 claim 1: 05-mixer, claimed +0.0115, verified +0.0118, replicated, record True
- pilot-02 claim 2: 05-mixer, claimed +0.0119, verified +0.0122, replicated, record True
