# odometer agent series: results

| Measure | Value |
|---|---|
| Attempts counted | 6 of 6 started |
| Attempts excluded (infrastructure, no claim) | 0 |
| Attempts waiting for verify_claims | 0 |
| Counted attempts with no claim | 0 |
| Counted attempts with a denied tool | 0 |
| Claims | 7: 7 replicated, 0 refuted, 0 inconclusive, 0 rejected |
| Survival per claim (primary) | 7/7 (Wilson 95% 0.646 to 1.000) |
| Survival per attempt (primary) | 6/6 (Wilson 95% 0.610 to 1.000) |
| Median shrinkage g/c, claims with c > 0 | +0.996 (n=7, 0 claimed no gain) |
| Records | 5 |
| Seed-runs before a claim, median (range) | 16 (8 to 46) |
| Best verified gain, bpb | +0.0188 |
| Survival from record 06-mixer-fold | 1/1 (Wilson 95% 0.207 to 1.000) |
| Survival from record 07-mixer-cold | 1/1 (Wilson 95% 0.207 to 1.000) |
| Survival from record 08-mixer-gate | 2/2 (Wilson 95% 0.342 to 1.000) |
| Survival from record 09-mixer-words | 1/1 (Wilson 95% 0.207 to 1.000) |
| Survival from record 10-mixer-fresh | 2/2 (Wilson 95% 0.342 to 1.000) |
| Null control on 06-mixer-fold | 8 nulls: false replicated 0/8 (Wilson 95% 0.000 to 0.324); false refuted 0; stage 1 passes 0; batch against batch 0/56 (Wilson 95% 0.000 to 0.064); stored batch offset +0.75 SE |
| Null control on 07-mixer-cold | 8 nulls: false replicated 0/8 (Wilson 95% 0.000 to 0.324); false refuted 0; stage 1 passes 0; batch against batch 0/56 (Wilson 95% 0.000 to 0.064); stored batch offset -0.03 SE |
| Null control on 08-mixer-gate | 8 nulls: false replicated 0/8 (Wilson 95% 0.000 to 0.324); false refuted 1; stage 1 passes 0; batch against batch 0/56 (Wilson 95% 0.000 to 0.064); stored batch offset -2.08 SE |
| Null control on 09-mixer-words | 8 nulls: false replicated 0/8 (Wilson 95% 0.000 to 0.324); false refuted 0; stage 1 passes 0; batch against batch 0/56 (Wilson 95% 0.000 to 0.064); stored batch offset +0.41 SE |
| Null control on 10-mixer-fresh | 8 nulls: false replicated 0/8 (Wilson 95% 0.000 to 0.324); false refuted 0; stage 1 passes 0; batch against batch 0/56 (Wilson 95% 0.000 to 0.064); stored batch offset -0.56 SE |

| Attempt | Claim | Base | Claimed gain | Verified gain | 95% half-width | p | Outcome | Record | Seed-runs before |
|---|---|---|---|---|---|---|---|---|---|
| attempt-01 | 1 | 06-mixer-fold | +0.0184 | +0.0185 | +0.0011 | +0.000 | replicated | True | 16 |
| attempt-02 | 1 | 07-mixer-cold | +0.0172 | +0.0188 | +0.0006 | +0.000 | replicated | True | 8 |
| attempt-03 | 1 | 08-mixer-gate | +0.0026 | +0.0025 | +0.0008 | +0.000 | replicated | False | 13 |
| attempt-04 | 1 | 08-mixer-gate | +0.0091 | +0.0095 | +0.0008 | +0.000 | replicated | True | 44 |
| attempt-05 | 1 | 09-mixer-words | +0.0055 | +0.0047 | +0.0013 | +0.000 | replicated | True | 16 |
| attempt-06 | 1 | 10-mixer-fresh | +0.0037 | +0.0034 | +0.0008 | +0.000 | replicated | False | 34 |
| attempt-06 | 2 | 10-mixer-fresh | +0.0041 | +0.0041 | +0.0009 | +0.000 | replicated | True | 46 |

Pilots ran before the series was registered and count toward nothing above.

- pilot-01 claim 1: 05-mixer, claimed +0.0009, verified +0.0004, inconclusive, record False
- pilot-02 claim 1: 05-mixer, claimed +0.0115, verified +0.0118, replicated, record True
- pilot-02 claim 2: 05-mixer, claimed +0.0119, verified +0.0122, replicated, record True
