"""Byte streams for training and held-out evaluation."""

import pathlib as _pathlib

import numpy as _np

from . import _engine

DATA = _pathlib.Path(__file__).resolve().parent.parent / "data"
HELDOUT_SPLITS = {"dev": 0, "test": 1}


class TrainData:
    """What an entry's train() receives. batch() is the only way to see training bytes."""

    def __init__(self, path=None):
        self._a = _np.fromfile(path or DATA / "train.bin", dtype=_np.uint8)
        self._starts = _np.flatnonzero(self._a[:-1] == 0) + 1

    @property
    def num_bytes(self):
        return int(self._a.size)

    def batch(self, batch_size, seq_len, align=False):
        """Random windows: x is (batch_size, seq_len) bytes, y is x shifted left by one.

        With align=True every window starts at the beginning of a story. Positions come from the verifier's
        seed, not the entry's.
        """
        B, T = int(batch_size), int(seq_len)
        if B < 1 or T < 1 or T >= self._a.size // 2:
            raise _engine.RuleViolation("bad batch shape")
        _engine._train_only("batch()")
        _engine._tick(16 * B + B * T)
        rng = _engine._M.data_rng
        if align:
            pos = self._starts[rng.integers(0, self._starts.size, size=B)]
            pos = _np.minimum(pos, self._a.size - T - 1)
        else:
            pos = rng.integers(0, self._a.size - T - 1, size=B)
        w = self._a[pos[:, None] + _np.arange(T + 1)].astype(_np.int32)
        return _engine.Tensor(w[:, :-1]), _engine.Tensor(w[:, 1:])


def heldout_windows(split, context, count):
    """The fixed evaluation windows for a held-out split, as a (count, context + 1) uint8 array.

    The held-out file is cut into non-overlapping windows; a fixed permutation assigns half to dev and half to
    test. Every entry is scored on the same bytes, so differences between entries are not eval-sampling noise.
    """
    a = _np.fromfile(DATA / "heldout.bin", dtype=_np.uint8)
    n = a.size // (context + 1)
    order = _np.random.default_rng(20261101).permutation(n)
    half = order[HELDOUT_SPLITS[split] :: 2][:count]
    return a[: n * (context + 1)].reshape(n, context + 1)[_np.sort(half)]
