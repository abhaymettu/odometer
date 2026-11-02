# Submitting a record

1. Write `entries/<yours>/train.py` with `train(data)` and `forward(params, x)`. Copy a recipe from `entries/` to start. `python verify.py entries/<yours>/train.py --smoke --split dev` checks in seconds that it runs and follows the rules.
2. Tune on dev, never on test: `python verify.py entries/<yours>/train.py --split dev --seeds 8`. Keep a log of what you tried. `entries/TUNING.md` shows the level of detail that helps a reviewer.
3. When it beats the dev mean of `records/current.json`, run the record rule: `python verify.py entries/<yours>/train.py --against records/current.json`. It tests on 8 fresh test seeds, and if that passes, again on 8 new ones. The log says `NEW RECORD` or `not a record`, and goes to `runs/`.
4. Open a pull request with the entry and that log. Someone other than you reruns the rule in CI (`.github/workflows/verify.yml`) and writes the review: what buys the gain, whether it abuses a price or a rule, and whether the number stands clear of your selection. `records/*/review.md` shows what a review covers.
5. `python -m study.ladder NN-name entries/<yours>/train.py --review review.md --verdict <log> --source "<who>"` seats it, measuring it on 8 dev and 8 test seeds that decide nothing else. Those stored seeds are what the next challenger has to beat.

Agents and people go through the same gate. If you used an agent, say which one and what it was given. If an attempt comes back `not a record`, that's useful too, so please file it.
