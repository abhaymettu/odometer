# For agents working in this repo

odometer is a training track. `verify.py` is the referee and `odometer/_engine.py` the metered engine. An entry is one file, `train.py`.

Commands, from the repo root with Python 3.12 and `numpy==2.5.4`:

- `python prepare.py` fetches the data and checks its hashes.
- `python verify.py path/to/train.py --smoke` checks in seconds that an entry runs.
- `python verify.py path/to/train.py --split dev --seeds 8` scores it on dev.
- `python -m pytest tests -q` runs the engine, attack and study tests.
- `python tools/build_page.py && python tools/figures.py` rebuilds the page's data and figures from the logs.

Rules that matter:

- Do not edit `verify.py` or `odometer/`. Records carry `code_sha256`, and any change to those files makes every stored record unverifiable. A changed track needs a new version and a new table.
- Tune on dev. The test split is for the record rule, and a stored outcome is final.
- Nothing on the page or in the README is typed by hand if a log has it. Change the log's source, then rebuild.
- Everything in `study/attempts/` is a record of what happened. Do not edit it.
