# Tuning log for the baseline entries

Every config I scored, with its dev mean and seed count, so a reader can see how much tuning went into each number. All scores are `verify.py --split dev` on fresh seeds at the real budget of 4e10 ticks, on one 4-vCPU VM, normally 4 seeds at a time. No full-budget run read the test split; two smoke runs did by mistake, and "How to read these numbers" below has the details and the overlaps. I made no change to the engine, its prices or the verifier.

The mixer wins because this corpus rewards long exact contexts, and the price list sells them cheaply: a thin hashed row costs about 140 ticks per byte, so it sees seven times the transformers' bytes and reaches back 12 bytes. "Why the mixer wins" below has the evidence.


## Summary, 8 fresh seeds on dev

| recipe | dev bpb | 95% CI | seconds per seed (train) | ticks/byte at eval | budget used |
|---|---|---|---|---|---|
| mixer | 1.1566 | 0.0009 | 96 | 11,153 | 0.9989 |
| ppm | 1.2692 | 0.0003 | 71 | 24,082 | 0.9998 |
| modern | 1.5080 | 0.0047 | 118 | 14,578 | 0.9989 |
| rnn | 1.5407 | 0.0062 | 44 | 3,191 | 0.9998 |
| gpt | 1.7171 | 0.0099 | 112 | 16,255 | 0.9995 |

## Every config tried

Seed-runs in the sweep log: 312. Outside the sweep: 5 single trainings kept for eval-only tests, 5 budget runs (one lost), 4 seeds of `r4-lr6` lost when the VM stopped, and 56 seed-runs of 8-seed finals (the 5 in `dev-8.json`, plus 2 superseded by reruns), for 382 of the 500 or so allowed. I stopped because the families plateaued, not because the budget ran out. Smoke runs at 2e9 ticks are not counted. A void row with nothing run is listed but not counted. Bpb is the dev mean over n fresh seeds. Rows are in the order they ran. Later rows inherit the code at that time, so a row is comparable with its neighbours rather than with the whole table.

### gpt

| what changed | n | dev bpb | takeaway |
|---|---|---|---|
| `gpt-base`: base | 2 | 1.9011 | Baseline with cheaper spelling of the same maths. |
| `gpt-b32-lr2`: BATCH 32, LR 0.002 | 2 | 2.1060 | Bigger batch, lower rate: fewer steps, much worse. Steps bind, not bytes. |
| `gpt-b8-lr5`: BATCH 8, LR 0.005 | 2 | 1.8348 | More steps and a higher rate: better than base. |
| `gpt-d64l2-b8`: DIM 64, LAYERS 2, BATCH 8, LR 0.005 | 2 | 1.7813 | Small GPT, high rate: better than three layers. |
| `gpt-d96l2-b16`: DIM 96, LAYERS 2, BATCH 16, LR 0.004 | 2 | 1.7685 | Two layers at d96: best GPT so far. |
| `gpt-d128l2-lr4`: DIM 128, LAYERS 2, LR 0.004 | 2 | 1.8020 | GPT d128 loses to d96. |
| `gpt-d96l2-lr6`: LAYERS 2, LR 0.006 | 2 | 1.7499 | GPT rate 6e-3 better than 4e-3. |
| `gpt-d96l2-untied`: LAYERS 2, LR 0.004, TIED 0 | 2 | 1.7308 | Untied head better than nanoGPT's tying here: tying forces the 96-dim table to do two jobs. |
| `g2-b8`: LAYERS 2, LR 0.006, TIED 0, BATCH 8 | 2 | 1.7660 | Batch 8 worse again. |
| `g2-lr6`: LAYERS 2, LR 0.006, TIED 0 | 2 | 1.7311 | Same as 4e-3 once untied. |
| `g2-h2`: LAYERS 2, LR 0.004, TIED 0, HEADS 2 | 2 | 1.7550 | Two heads lose to four. |
| `g2-t48`: LAYERS 2, LR 0.004, TIED 0, BLOCK 48, EVAL_BLOCK 48, STRIDE 16 | 2 | 1.7310 | Block 48 with stride 16: same score as the base, with cheaper attention. |
| `g2-mlp2`: LAYERS 2, LR 0.004, TIED 0, MLP 2 | 2 | 1.7400 | MLP 2 close to 4. |
| `g2-beta95`: LAYERS 2, LR 0.004, TIED 0, BETA2 0.95 | 2 | 1.7522 | beta2 0.95 no better. |
| `g3-t48-lr6`: LAYERS 2, LR 0.006, TIED 0, BLOCK 48, EVAL_BLOCK 48, STRIDE 16 | 2 | 1.7091 | Block 48 at 6e-3: best GPT. |
| `g3-t48-lr8`: LAYERS 2, LR 0.008, TIED 0, BLOCK 48, EVAL_BLOCK 48, STRIDE 16 | 2 | 1.7174 | 8e-3: slightly worse. |
| `g3-t32`: LAYERS 2, LR 0.006, TIED 0, BLOCK 32, EVAL_BLOCK 32, STRIDE 8, BATCH 24 | 2 | 1.7181 | Block 32: worse. |
| `g3-d128`: LAYERS 2, LR 0.005, TIED 0, BLOCK 48, EVAL_BLOCK 48, STRIDE 16, DIM 128 | 2 | 1.7592 | d128: worse. |

### modern

| what changed | n | dev bpb | takeaway |
|---|---|---|---|
| `modern-base`: base | 2 | 1.7128 | First port at d128 L3 is already far ahead of the GPT; Muon's few steps go a long way. |
| `modern-noqk`: QKNORM 0 | 2 | 2.0368 | Without QK-norm: 0.32 worse. QK-norm stays. |
| `modern-b8`: BATCH 8 | 2 | 1.8011 | Batch 8 is too noisy for Muon. |
| `modern-d96h2l2`: DIM 96, LAYERS 2 | 2 | 1.6118 | Smaller and shallower wins: 1.61. |
| `modern-ns1`: NS_STEPS 1 | 2 | 1.8556 | One Newton-Schulz step is not enough. |
| `modern-lr05`: LR_MUON 0.05 | 2 | 1.7506 | Muon 0.05 worse than the base rate at d128 L3. |
| `modern-t32`: BLOCK 32, BATCH 32, EVAL_BLOCK 64 | 2 | 4.9392 | Mistake: eval block 64 with positions trained to 32 only. Result is meaningless. |
| `modern-d96l1`: DIM 96, LAYERS 1 | 2 | 1.8087 | One layer loses. |
| `modern-d64l2`: DIM 64, HEADS 1, LAYERS 2 | 2 | 1.6850 | d64 with one head loses to d96. |
| `modern-d96l2-rot`: DIM 96, LAYERS 2, ROTARY 1 | 2 | 1.5669 | Rotary beats learned positions by 0.04 at the same size. Rotary stays. |
| `modern-d128l2`: DIM 128, LAYERS 2 | 2 | 1.6537 | Wider costs more than it buys. |
| `m-rot-b8`: DIM 96, LAYERS 2, ROTARY 1, BATCH 8 | 2 | 1.6076 | Rotary at batch 8: worse, the noise costs more than the extra steps pay. |
| `m-rot-b24`: DIM 96, LAYERS 2, ROTARY 1, BATCH 24 | 2 | 1.5469 | Batch 24 better than 16. |
| `m-rot-lr05`: DIM 96, LAYERS 2, ROTARY 1, LR_MUON 0.05 | 2 | 1.5836 | Muon 0.05 too high once rotary is in. |
| `m-rot-lr02`: DIM 96, LAYERS 2, ROTARY 1, LR_MUON 0.02 | 2 | 1.5510 | Muon 0.02 better than 0.03. |
| `m-rot-d128`: DIM 128, LAYERS 2, ROTARY 1 | 2 | 1.5808 | Wider still loses. |
| `m-rot-l3`: DIM 96, LAYERS 3, ROTARY 1 | 2 | 1.5377 | Third layer pays slightly. |
| `m-rot-h4`: DIM 96, LAYERS 2, ROTARY 1, HEADS 4 | 2 | 1.5400 | Four heads pay slightly. |
| `m-rot-t96`: DIM 96, LAYERS 2, ROTARY 1, BLOCK 96, BATCH 11, EVAL_BLOCK 96, STRIDE 48 | 2 | void | Void: forward() returned the wrong length, a bug in my stride loop, since fixed. Ran train and counted. |
| `m2-l3-h4`: DIM 96, LAYERS 3, ROTARY 1, HEADS 4 | 2 | 1.5506 | Three layers and four heads together do not add up. |
| `m2-l3-b24`: DIM 96, LAYERS 3, ROTARY 1, BATCH 24 | 2 | 1.5311 | L3, batch 24: 1.53. |
| `m2-h4-b24-lr2`: DIM 96, LAYERS 2, ROTARY 1, HEADS 4, BATCH 24, LR_MUON 0.02 | 2 | 1.5251 | L2, four heads, batch 24, Muon 0.02: 1.525. |
| `m2-ev128`: DIM 96, LAYERS 2, ROTARY 1, EVAL_BLOCK 128 | 2 | 2.7960 | Eval block 128 on a model trained at 64: rotary does not extrapolate here. Much worse and unstable. |
| `m2-mlp4`: DIM 96, LAYERS 2, ROTARY 1, MLP 4 | 2 | 1.5549 | MLP 4 costs more than it buys. |
| `m2-adam6`: DIM 96, LAYERS 2, ROTARY 1, LR_ADAM 0.006 | 2 | 1.5508 | Adam 6e-3 on the embedding and head no better. |
| `m3-ns5`: DIM 96, LAYERS 2, ROTARY 1, HEADS 4, BATCH 24, LR_MUON 0.02, NS_STEPS 5 | 2 | 1.5313 | Five Newton-Schulz steps: no better than three, and more ticks. |
| `m3-ns2`: DIM 96, LAYERS 2, ROTARY 1, HEADS 4, BATCH 24, LR_MUON 0.02, NS_STEPS 2 | 2 | 1.5726 | Two steps: 0.05 worse. Three is the knee. |
| `m3-l3`: DIM 96, LAYERS 3, ROTARY 1, HEADS 4, BATCH 24, LR_MUON 0.02 | 2 | 1.5246 | Third layer: same within noise. |
| `m3-d128`: DIM 128, LAYERS 2, ROTARY 1, HEADS 4, BATCH 24, LR_MUON 0.02 | 2 | 1.5330 | d128: same within noise, more ticks. |
| `m3-ramp`: DIM 96, LAYERS 2, ROTARY 1, HEADS 4, BATCH 24, LR_MUON 0.02, RAMP_UP 0.05, RAMP_DOWN 0.5 | 2 | 1.5265 | Shorter ramp up: same. |
| `m3-t48`: DIM 96, LAYERS 2, ROTARY 1, HEADS 4, BATCH 32, LR_MUON 0.02, BLOCK 48, EVAL_BLOCK 48, S... | 2 | 1.5207 | Block 48 at batch 32: best modern. |
| `m3-lr15`: DIM 96, LAYERS 2, ROTARY 1, HEADS 4, BATCH 24, LR_MUON 0.015 | 2 | 1.5329 | Muon 0.015: slightly worse. |
| `m3-mlp3`: DIM 96, LAYERS 2, ROTARY 1, HEADS 4, BATCH 24, LR_MUON 0.02, MLP 3 | 2 | 1.5130 | MLP 3: best modern row, 0.012 better than its base. I misread this at the time and left MLP at 2; see m4-*. |
| `m4-mlp3`: MLP 3 | 4 | 1.5130 | MLP 3 on the final config, n=4: 0.011 better than the MLP 2 final. Chosen. |
| `m4-mlp3-b24t64`: MLP 3, BATCH 24, BLOCK 64, EVAL_BLOCK 64 | 4 | 1.5190 | MLP 3 with block 64 and batch 24, the m3-mlp3 config: block 48 still wins. |
| `m4-mlp4`: MLP 4 | 4 | 1.5147 | MLP 4: within noise of 3, with more spread and more ticks per byte. |

### rnn

| what changed | n | dev bpb | takeaway |
|---|---|---|---|
| `rnn-base`: base | 2 | 2.4371 | d192 too wide for Adam at this step count; loss still high at the end. |
| `rnn-d128`: DIM 128, EMBED 32 | 2 | 1.8657 | Narrower GRU gets more updates and beats the wider one by 0.57. |
| `rnn-d128-lr2`: DIM 128, EMBED 32, LR 0.002 | 2 | 2.0803 | Lower rate hurts: too few steps for 2e-3. |
| `rnn-d256-lr1`: DIM 256, EMBED 64, LR 0.001 | 2 | 2.9242 | Wide and slow: far from converged. |
| `rnn-d128-b32`: DIM 128, EMBED 32, BATCH 32 | 2 | 1.7748 | Half the batch, twice the updates: better. |
| `rnn-d96-lr5`: DIM 96, EMBED 32, LR 0.005 | 2 | 1.7814 | d96 at 5e-3 is close to d128. |
| `rnn-d96-b32`: DIM 96, EMBED 32, BATCH 32 | 2 | 1.7975 | d96 slightly behind d128. |
| `rnn-d128-b16`: DIM 128, EMBED 32, BATCH 16 | 2 | 1.7436 | Batch 16 better than 32 at TBPTT 32. |
| `rnn-d128-b32-t16`: DIM 128, EMBED 32, BATCH 32, TBPTT 16 | 2 | 1.6376 | TBPTT 16: twice the updates, 0.14 better. |
| `rnn-d96-2l-b32`: DIM 96, EMBED 32, BATCH 32, LAYERS 2 | 2 | 2.9689 | Two layers without norm: unstable, mean near 3 bpb. |
| `r2-t8`: DIM 128, EMBED 32, BATCH 32, TBPTT 8 | 2 | 1.5814 | TBPTT 8 better than 16 at 3e-3. |
| `r2-d192`: DIM 192, EMBED 48, BATCH 32, TBPTT 16 | 2 | 1.6064 | d192 at TBPTT 16 now competitive. |
| `r2-b16`: DIM 128, EMBED 32, BATCH 16, TBPTT 16 | 2 | 1.6464 | Batch 16 no better than 32 at TBPTT 16. |
| `r2-lr5`: DIM 128, EMBED 32, BATCH 32, TBPTT 16, LR 0.005 | 2 | 1.5472 | Rate 5e-3 at TBPTT 16: 1.55. |
| `r2-w128`: DIM 128, EMBED 32, BATCH 32, TBPTT 16, WINDOW 128 | 2 | 1.6817 | Short windows cut the state too often. |
| `r2-clip5`: DIM 128, EMBED 32, BATCH 32, TBPTT 16, CLIP 5.0 | 2 | 1.6244 | A looser clip helps a little. |
| `r3-t8`: DIM 128, EMBED 32, BATCH 32, TBPTT 8, LR 0.005 | 2 | 1.5569 | TBPTT 8 at 5e-3: slightly worse than 16. |
| `r3-lr8`: DIM 128, EMBED 32, BATCH 32, TBPTT 16, LR 0.008 | 2 | 1.5337 | Rate 8e-3: best GRU. |
| `r3-d192`: DIM 192, EMBED 48, BATCH 32, TBPTT 16, LR 0.005 | 2 | 1.5499 | d192: no better, more seed spread. |
| `r3-e64`: DIM 128, EMBED 64, BATCH 32, TBPTT 16, LR 0.005 | 2 | 1.5568 | Embed 64: no better. |
| `r3-clip3`: DIM 128, EMBED 32, BATCH 32, TBPTT 16, LR 0.005, CLIP 3.0 | 2 | 1.5436 | Clip 3: same. |
| `r3-b48`: DIM 128, EMBED 32, BATCH 48, TBPTT 16, LR 0.005 | 2 | 1.5443 | Batch 48: same. |
| `r4-lr5`: LR 0.005 | 4 | 1.5396 | Rate 5e-3 on the final GRU: spread a third of the 6e-3 row. Chosen. |
| `r4-lr6`: LR 0.006 | 4 | 1.5380 | Rate 6e-3: mean level with 5e-3, three times the seed spread. |
| `r4-lr4`: LR 0.004 | 4 | 1.5788 | Rate 4e-3: 0.04 worse. The low side of the curve is steep. |

### mixer

| what changed | n | dev bpb | takeaway |
|---|---|---|---|
| `mixer-base`: base | 2 | 1.3952 | First try at the neural mixer: 0.17 better than anything else so far, seed SD 0.0001. |
| `mx-lrt06`: LR_TABLE 0.06 | 2 | 1.3621 | Table rate up: large gain. |
| `mx-lrt015`: LR_TABLE 0.015 | 2 | 1.4552 | Table rate down: loses 0.06. |
| `mx-e32`: EMBED  32,  ROWS  [256,  8191,  16381,  32749,  32749,  32749] | 2 | void | Void: my sweep passed a list, which the static rules reject. Nothing ran; not counted. |
| `mx-e8`: EMBED 8 | 2 | 1.4042 | EMBED 8 close to 16 at half the backward price. |
| `mx-h128`: HIDDEN 128 | 2 | 1.3904 | Hidden 128 close to 256 and cheaper. |
| `mx-b64`: BATCH 64 | 2 | 1.3861 | Batch 64 at the old rate is a small win. |
| `mx-o8`: ORDERS  [1,  2,  3,  4,  5,  6,  8],  ROWS  [256,  8191,  32749,  65521,  65521,  65521... | 2 | void | Void: same list bug. Nothing ran; not counted. |
| `mx-b16`: BATCH 16 | 2 | 1.4201 | Batch 16 worse: the table update is dense, so small batches pay for it. |
| `mx-lrt12`: LR_TABLE 0.12 | 2 | 1.3456 | Table rate keeps helping. |
| `mx-lrt25`: LR_TABLE 0.25 | 2 | 1.3364 | Table rate 0.25: near the top of the curve. |
| `mx-e32-lrt06`: EMBED 32, ROWS [256, 8191, 16381, 32749, 32749, 32749], LR_TABLE 0.06 | 2 | 1.3691 | EMBED 32 no better than 16; twice the backward price. |
| `mx-o8-lrt06`: ORDERS [1, 2, 3, 4, 5, 6, 8], ROWS [256, 8191, 32749, 65521, 65521, 65521, 65521], LR_T... | 2 | void | Void: over the parameter cap. |
| `mx-h128-lrt06`: HIDDEN 128, LR_TABLE 0.06 | 2 | 1.3587 | Hidden 128 beats 256 at the same rate. |
| `mx-b64-lrt12`: BATCH 64, LR_TABLE 0.12 | 2 | 1.3359 | Batch 64 with rate 0.12. |
| `mx-b128-lrt25`: BATCH 128, LR_TABLE 0.25 | 2 | 1.3244 | Batch 128 about the same as 64 and cheaper per byte. |
| `mx-b64-lrt25`: BATCH 64, LR_TABLE 0.25 | 2 | 1.3265 | Batch 64 at 0.25. |
| `mx-b64-lrt5`: BATCH 64, LR_TABLE 0.5 | 2 | 1.3230 | Rate 0.5 about the same as 0.25. |
| `mx-o8s-b64-lrt25`: BATCH 64, LR_TABLE 0.25, ORDERS [1, 2, 3, 4, 5, 6, 8], ROWS [256, 4093, 16381, 32749, 6... | 2 | 1.2571 | Adding order 8 with smaller tables: 0.07 better. Reach beats table size. |
| `mx-o12s-b64-lrt25`: BATCH 64, LR_TABLE 0.25, ORDERS [1, 2, 3, 4, 5, 6, 8, 12], ROWS [256, 4093, 16381, 3274... | 2 | 1.2444 | Adding order 12: another 0.013. |
| `mx-lrm6-b64-lrt25`: BATCH 64, LR_TABLE 0.25, LR_MIX 0.006 | 2 | 1.3238 | Mixer rate 6e-3 about equal. |
| `mx2-o12`: BATCH 128, HIDDEN 128, LR_TABLE 0.25, ORDERS [1, 2, 3, 4, 5, 6, 7, 8, 10, 12], ROWS [25... | 2 | 1.2276 | Ten orders to 12: best so far. |
| `mx2-base`: BATCH 128, HIDDEN 128, LR_TABLE 0.25, ORDERS [1, 2, 3, 4, 5, 6, 8], ROWS [256, 4093, 16... | 2 | 1.2461 | New base: orders to 8, batch 128, hidden 128. |
| `mx2-word3`: BATCH 128, HIDDEN 128, LR_TABLE 0.25, ORDERS [1, 2, 3, 4, 5, 6, 8], ROWS [256, 4093, 16... | 2 | 1.2363 | Three word contexts help a little at this point. |
| `mx2-word2`: BATCH 128, HIDDEN 128, LR_TABLE 0.25, ORDERS [1, 2, 3, 4, 5, 6, 8], ROWS [256, 4093, 16... | 2 | 1.2483 | Word contexts (two) do not pay after shrinking tables to fit them. |
| `mx2-h256`: BATCH 128, HIDDEN 256, LR_TABLE 0.25, ORDERS [1, 2, 3, 4, 5, 6, 8], ROWS [256, 4093, 16... | 2 | 1.2547 | Hidden 256 loses. |
| `mx2-e8`: BATCH 128, HIDDEN 128, LR_TABLE 0.25, ORDERS [1, 2, 3, 4, 5, 6, 8], ROWS [256, 8191, 32... | 2 | 1.2438 | EMBED 8 with double rows beats EMBED 16. |
| `mx2-h64`: BATCH 128, HIDDEN 64, LR_TABLE 0.25, ORDERS [1, 2, 3, 4, 5, 6, 8], ROWS [256, 4093, 163... | 2 | 1.2425 | Hidden 64 wins: fewer ticks per byte. |
| `mx2-lrm6`: BATCH 128, HIDDEN 128, LR_TABLE 0.25, ORDERS [1, 2, 3, 4, 5, 6, 8], ROWS [256, 4093, 16... | 2 | 1.2408 | Mixer rate 6e-3 wins. |
| `mx3-base`: base | 2 | 1.2217 | Zero-initialized tables with a bias, history pad: same as before, cleaner for unseen rows. |
| `mx3-flat`: TABLE_DECAY 0 | 2 | 1.2125 | Flat table rate: AdaGrad already decays; 0.01 better. |
| `mx3-flat-lrt1`: TABLE_DECAY 0, LR_TABLE 0.1 | 2 | 1.2237 | Flat at 0.1 is too low. |
| `mx3-e8`: EMBED 8, ROWS [256, 8191, 16381, 32749, 65521, 65521, 65521, 65521, 65521, 65521] | 2 | 1.2091 | EMBED 8: best so far. |
| `mx3-o16`: ORDERS [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16], ROWS [256, 4093, 16381, 32749, 32749, 3274... | 2 | 1.2280 | Order 16 with smaller tables loses. |
| `mx3-t32`: WINDOW 32, BATCH 256 | 2 | 1.2218 | Window 32 same as 64. |
| `mx3-word3`: ROWS [256, 4093, 8191, 16381, 32749, 32749, 32749, 32749, 16381, 16381], WORD_ROWS [163... | 2 | 1.2306 | Word contexts no longer pay. |
| `mx3-h64`: HIDDEN 64 | 2 | 1.2182 | Hidden 64 small win. |
| `mx4-e8-flat`: EMBED 8, ROWS [256, 8191, 16381, 32749, 65521, 65521, 65521, 65521, 65521, 65521], TABL... | 2 | 1.1982 | EMBED 8 and flat rate add up: 1.198. |
| `mx4-e8-flat-h64`: EMBED 8, ROWS [256, 8191, 16381, 32749, 65521, 65521, 65521, 65521, 65521, 65521], TABL... | 2 | 1.1957 | Hidden 64: 1.196. |
| `mx4-e8-flat-b256`: EMBED 8, ROWS [256, 8191, 16381, 32749, 65521, 65521, 65521, 65521, 65521, 65521], TABL... | 2 | 1.2021 | Batch 256 slightly worse. |
| `mx4-e8-flat-lrt5`: EMBED 8, ROWS [256, 8191, 16381, 32749, 65521, 65521, 65521, 65521, 65521, 65521], TABL... | 2 | 1.1967 | Rate 0.5 same as 0.25. |
| `mx4-e8-flat-word3`: EMBED 8, ROWS [256, 4093, 8191, 16381, 32749, 65521, 65521, 65521, 65521, 65521], TABLE... | 2 | 1.1978 | Word contexts: zero gain. Removed from the entry. |
| `mx4-e4-flat`: EMBED 4, ROWS [256, 8191, 32749, 65521, 131071, 131071, 131071, 131071, 131071, 131071]... | 2 | 1.2076 | EMBED 4 too thin. |
| `mx4-e8-flat-o14`: EMBED 8, ROWS [256, 4093, 16381, 32749, 65521, 65521, 65521, 32749, 32749, 32749, 32749... | 2 | 1.2034 | Twelve orders to 14: table size hurts more than reach helps. |
| `mx5-base`: base | 2 | 1.1946 | Dropped word code, adds eps at init: same. |
| `mx5-every2`: TABLE_EVERY 2 | 2 | 1.1897 | Table update every 2 batches: 0.005 better. |
| `mx5-every4`: TABLE_EVERY 4, BATCH 64 | 2 | 1.1886 | Every 4 at batch 64: table updates are dense, so spacing them buys bytes. |
| `mx5-lrm10`: LR_MIX 0.01 | 2 | 1.1917 | Mixer rate 1e-2: small win. |
| `mx5-relu`: SQUARE 0 | 2 | 1.2076 | Plain ReLU: 0.013 worse than squared. |
| `mx5-e6`: EMBED 6, ROWS [256, 8191, 16381, 32749, 87359, 87359, 87359, 87359, 87359, 87359] | 2 | 1.1970 | EMBED 6 with more rows: no better. |
| `mx5-b64`: BATCH 64 | 2 | 1.2014 | Batch 64 with every-batch table updates: worse. |
| `mx5-h32`: HIDDEN 32 | 2 | 1.2004 | Hidden 32: worse. |
| `mx6-rows76k`: BATCH 64, TABLE_EVERY 4, ROWS [256, 8191, 16381, 32749, 75989, 75989, 75989, 75989, 759... | 2 | 1.1580 | All 4 M floats used: 1.158. |
| `mx6-every4`: BATCH 64, TABLE_EVERY 4 | 2 | 1.1598 | With window votes at eval: 1.160, 0.03 better than without. |
| `mx6-embed`: BATCH 64, TABLE_EVERY 4, EMBED [16, 16, 16, 12, 8, 8, 8, 8, 8, 8], ROWS [256, 4093, 819... | 2 | 1.1619 | Wider rows on low orders: no gain. |
| `mx6-dup`: BATCH 64, TABLE_EVERY 4, ORDERS [1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 10, 12], ROWS [256, 8191... | 2 | 1.1605 | Two hashes per order for orders 6 and 8: no gain. |
| `mx6-lrt35`: BATCH 64, TABLE_EVERY 4, LR_TABLE 0.35 | 2 | 1.1604 | Table rate 0.35: same. |
| `mx6-every8`: BATCH 32, TABLE_EVERY 8 | 2 | 1.1643 | Every 8 at batch 32: worse. |
| `mx6-nomom`: BATCH 64, TABLE_EVERY 4, LR_MIX 0.003 | 2 | 1.1624 | Mixer rate 3e-3: worse. |
| `mx7-second`: SECOND 1 | 2 | 1.1624 | A second hidden layer: no gain. |
| `mx7-direct`: DIRECT 1 | 2 | 1.1608 | A direct path from rows to logits: no gain. |
| `mx7-drop7`: ORDERS [1, 2, 3, 4, 5, 6, 8, 10, 12], ROWS [256, 8191, 16381, 32749, 75989, 75989, 9999... | 2 | 1.1565 | Dropping order 7 and giving its floats to 8, 10, 12: 1.1565. |
| `mx7-o1469`: ORDERS [1, 2, 3, 4, 6, 8, 11], ROWS [256, 8191, 16381, 49999, 129419, 129419, 129419] | 2 | 1.1649 | Seven orders with big tables: worse. |
| `mx7-w32`: WINDOW 32, BATCH 128 | 2 | 1.1570 | Window 32 with batch 128: same. |
| `mx7-h96`: HIDDEN 96 | 2 | 1.1575 | Hidden 96: same. |

### ppm

| what changed | n | dev bpb | takeaway |
|---|---|---|---|
| `ppm-first`: base | 2 | 2.9841 | Count tables at high load: false hits destroy the chain. See the probes below. |
| `ppm-c1`: ORDERS [0, 1, 2, 3, 4, 5, 6, 7, 8, 10], NARROW [509, 4093, 32749, 131071, 262139, 52428... | 2 | 1.3637 | Lower load and fitted constants: real run 1.36. |
| `pp2-tau2x`: ALPHA [1.0, 8.0, 8000.0, 1200.0, 1000.0, 1600.0, 1600.0, 1600.0, 1600.0, 1600.0], TAU [... | 2 | 1.2839 | Prune twice as hard: loses real keys. |
| `pp2-base`: ALPHA [1.0, 8.0, 8000.0, 1200.0, 1000.0, 1600.0, 1600.0, 1600.0, 1600.0, 1600.0] | 2 | 1.2692 | New table sizing, window stages, fitted constants: 1.269. Seed SD 0.0001. |
| `pp2-tauhalf`: ALPHA [1.0, 8.0, 8000.0, 1200.0, 1000.0, 1600.0, 1600.0, 1600.0, 1600.0, 1600.0], TAU [... | 2 | 1.2725 | Prune half as hard: more false hits. |
| `pp2-o12`: ORDERS [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12], NARROW [509, 16381, 131071, 524287, 524287,... | 2 | 1.2876 | Order 12 with smaller tables: worse. The 4 M cap binds. |

### hybrid (dropped)

| what changed | n | dev bpb | takeaway |
|---|---|---|---|
| `hy-gru64`: BATCH 64, TABLE_EVERY 4 | 2 | 1.2094 | A GRU fed by the mixer: 0.05 worse. Its ticks cost more bytes than its memory earns. |
| `hy-gru32`: BATCH 64, TABLE_EVERY 4, GRU 32 | 2 | 1.2027 | Smaller GRU: still 0.04 worse. Hybrid dropped. |

## Budget still binds

One seed each at three tick budgets, with the same code and constants.

| recipe | 2e10 | 4e10 (8 seeds) | 8e10 |
|---|---|---|---|
| mixer | 1.1955 | 1.1566 | 1.1320 |
| ppm | 1.2956 | 1.2692 | 1.2801 |

The mixer keeps improving, so the budget is still doing work at 4e10. PPM gets worse at 8e10 with its constants held. My read is that TAU was fit at 4e10, so twice the counts lets more keys survive the fold, the tables load up and false hits grow. I did not run a rescaled TAU, so that explanation is unverified.

## What the price list rewards

Ticks per byte measured at 3e9 ticks, scaled to 4e10. The training file is 66.0 M bytes.

| recipe | ticks per byte | bytes seen | batches | params |
|---|---|---|---|---|
| gpt | 19878 | 2.0 M | 2620 | 0.28 M |
| modern | 19103 | 2.1 M | 1363 | 0.23 M |
| rnn | 10739 | 3.7 M | 455 windows | 0.10 M |
| mixer | 2582 | 15.5 M | 2751 | 4.10 M |
| ppm | 411 | 97.2 M | 67 | 3.82 M |

## Why the mixer wins

It sees about seven times the bytes of the transformers, and it spends its ticks on the one thing this corpus pays for, which is long exact contexts.

**The corpus repeats itself.** 85% of order-12 dev contexts occur in train. Exact unbounded counts, computed offline on the first 128 dev windows, get these bits per byte with contexts up to each order:

| up to order | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|
| bpb | 3.30 | 2.41 | 1.83 | 1.53 | 1.37 | 1.26 | 1.19 | 1.14 | 1.08 | 1.05 |

A two-layer model with a 48-byte window has to learn that memory into 0.2 M floats from 2 M bytes. A table just stores it.

**Reach is nearly free.** Backward through `embedding()` costs 16 ticks per float whatever the table size, and hashing costs about 2 ticks. A thin row of 8 floats therefore buys one more order for about 140 ticks per byte, against about 19000 for a whole transformer pass. Adding reach gave the largest drops in the log: orders to 6 scored 1.3265, adding order 8 gave 1.2571, adding order 12 gave 1.2444.

**Rows beat counts under the 4.19 M float cap.** PPM sees six times the bytes and needs no gradients, yet stops at 1.27. It probes about 100 candidate bytes per position per order, so the false hit rate is about the load times 100. Tuning its tables is mostly a fight with that number. My read of the gap: a trained row shares its 8 floats across every context that hashes there, and the hidden layer can learn to discount collisions, so it degrades gently where counts fall off a cliff.

**Window votes at eval are free.** Counts taken inside the 512-byte window, run as a PPM backoff after the softmax, took the first mixer from 1.1984 to 1.1691 for zero training ticks. They give the online adaptivity a compressor has, such as a character's name from earlier in the story.

**The logits are calibrated.** Dividing them by a temperature only hurts: 0.85 gives 1.1703, 0.9 gives 1.1623, 0.95 gives 1.1581, 1.05 gives 1.1588, against about 1.1571 at 1.0. So no cheap gain is left there.

The path to the number, from the rows below:

| step | row | dev bpb |
|---|---|---|
| first try | `mixer-base` | 1.3952 |
| table rate 0.25, batch 64 | `mx-b64-lrt25` | 1.3265 |
| add order 8 | `mx-o8s-b64-lrt25` | 1.2571 |
| ten orders to 12, batch 128, hidden 128 | `mx2-o12` | 1.2276 |
| flat table rate | `mx3-flat` | 1.2125 |
| EMBED 8 with twice the rows | `mx4-e8-flat` | 1.1982 |
| hidden 64 | `mx4-e8-flat-h64` | 1.1957 |
| table update every 4 batches | `mx5-every4` | 1.1886 |
| three vote stages at eval | `mx6-every4` | 1.1598 |
| use all the floats | `mx6-rows76k` | 1.1580 |
| drop order 7 | `mx7-drop7` | 1.1565 |

The best recent rows (`mx7-drop7`, `mx7-w32`, `mx7-h96`, `mx6-rows76k`) sit within 0.002 of each other at n=2, and the 8-seed SD is 0.0011, so these knobs have plateaued. That is why I stopped here with budget left.

The mixer's low seed spread matters for the record rule. `verify.py` flags a challenger whose seed SD is below 0.3 of the incumbent's, and the PPM's 0.0003 would sit just under that line against the mixer's 0.0011, for honest reasons: a count table has almost nothing random in it.

## Eval-only tests

These change `forward()` only, on parameters kept from one training, so they cost one training and no extra ticks. Each is one dev evaluation of one seed. The constants they chose were fit on dev, like everything else here.

First mixer, in-window vote stages, by order (none 1.1984):

| stages | 4 | 6 | 8 | 3,5 | 4,6 | 4,8 | 3,4,6 | four stages |
|---|---|---|---|---|---|---|---|---|
| bpb | 1.1772 | 1.1800 | 1.1888 | 1.1741 | 1.1725 | 1.1752 | 1.1691 | about 1.1687 |

On stages 3,4,6: discount 0.3 gives 1.1699, 0.8 gives 1.1718, twice the concentration gives 1.1731, half gives 1.1693.

Final mixer: base 1.1576, chosen concentrations (35,6,2) 1.1571, stages on orders 3 and 4 only 1.1625, votes weighted by recency with a half-life of 256 bytes 1.1665, with a half-life of 64 bytes 1.1788.

PPM: no in-window stages 1.3172, base 1.2812, twice the concentration 1.2696 (chosen), four times 1.2848, half 1.3243, other variants between 1.268 and 1.278.

## Offline probes

Numpy, run outside the verifier to decide what to build. They read dev windows and train counts directly, so none of these numbers is a verified score. The first 32 dev windows are easier than the next 32 by about 0.07, so compare within a table only, never against the sweep.

| exact counts, pruned, 128 windows | 36.6 M keys | 14.8 M | 7.6 M | 4.6 M |
|---|---|---|---|---|
| bpb | 1.048 | 1.071 | 1.103 | 1.134 |

| 2.06 M keys, 32 windows | exact | plain 4 M floats | two-hash min | signed sketch | plain 16 M floats |
|---|---|---|---|---|---|
| bpb | 1.1906 | 1.3086 | 1.2823 | 1.4151 | 1.2173 |

Logistic mix of chain and order logits on windows from the other half: exact chain 1.2674, logit 1.2589, both 1.2277. At 4 M floats, chain 1.3647, logit 1.347. In-window stages on exact tables moved 1.1347 to 1.1022. Check bits stored in the fraction of a float failed (1.6 to 4 bpb), because float rounding at high counts eats them.

## Tried outside the sweep log

Count stages on top of the mixer were catastrophic, 2.57 to 5.8 bpb, with no low-order guard. A debug run showed about 13 false candidates per position at load 0.11, which is false hits times candidates rather than a hash bug. Without those stages that hybrid scored 1.1864 to 1.1878, worse than the pure mixer, because splitting ticks and floats cost about 0.017. A GRU fed by the mixer also lost (`hy-gru64`, `hy-gru32`). `entries/hybrid` was deleted.

Skipped by reasoning, not run: a 128-symbol head saves about 576 ticks per byte, but I estimated the non-ASCII penalty at about 0.005, which cancels it. Two-hash PPM was only simulated (table above), and at about 2% it would not change the ranking.

I left window votes out of gpt, modern and rnn because they are not part of those named recipes. They gave the mixer about 2%, and I did not test them on the other three.

## How to read these numbers

- **Selection on dev.** More than 100 configs were picked between on the same dev split, and eval constants were fit on it. The finals use fresh seeds but the same bytes, so expect a test score somewhat worse than dev.
- **Rows at n=2** were enough to move in a direction, not to separate neighbours. Rows differ in code as well as constants, since the code kept changing.
- **Concurrency.** The limit was 4 seeds at once. Sweep jobs overlapped above that a few times because of a bug in my process count, and the last budget runs overlapped with `r4-lr5`. Scores are deterministic in ticks, so they are unaffected, but the seconds in those rows are unreliable and are not in this log. The finals ran alone, at 4 at a time.
- **Two finals were rerun.** `rnn` at LR 8e-3 scored 1.5535 +- 0.0146 on 8 seeds, well off its n=2 row of 1.5337, so I bracketed the rate (`r4-*`) and moved to 5e-3. `modern` with MLP 2 scored 1.5238 +- 0.0024 on 8 seeds. While writing this log I found `m3-mlp3` had scored 1.5130 and that I had read it wrong and left MLP at 2. `m4-*` confirmed it on the final config, so MLP is now 3. Both superseded runs count toward the seed-runs. `dev-8.json` holds the rerun.
- **Lost runs.** The VM stopped at 06:46 UTC, not by my hand, with `r4-lr6` (4 seeds) and one PPM 8e10 run in flight. Neither left a score. I resumed it and reran both on new seeds. The first launch of the first sweep job also died mid-run with no results, and I did not count it.
- **A slip on the test split.** My last two smoke runs on the Mac, for `modern` and `rnn`, left out `--split dev`, and `verify.py` defaults to test. That scored 8 seeds each on 32 test windows, at the smoke budget of 2e9 ticks: 2.96 and 3.83 bpb. No choice used them, since both configs were already picked and these are models trained on a twentieth of the budget. Every full-budget run and every other smoke run was on dev. Even so, the test split has been read twice, and a reader should know. Logs are `runs/modern-20261102-010104.json` and `runs/rnn-20261102-010108.json`, on the machine that ran them only, as `runs/` is ignored by git.

## Next ideas, untried

- A two-hash minimum on the PPM tables, simulated at about -2%.
- An SSE or APM stage after the mixer, as lpaq does, using the window counts as context.
- Seeding the mixer's rows from a counting pass, since a count pass costs a fraction of a training pass.
- Window votes on the transformers and the GRU.
- A rescaled TAU for PPM if the budget ever rises.
