"""Download the track's data once: TinyStories V2 (GPT-4 stories) as raw bytes.

Train: the first 64 MiB of TinyStoriesV2-GPT4-train.txt. Held out: TinyStoriesV2-GPT4-valid.txt, which the
verifier splits into a dev pool and a test pool. Story separators become one NUL byte. Nothing is redistributed;
the files come from the dataset's own Hugging Face page and are checked against fixed hashes.

    python prepare.py
"""

import hashlib
import pathlib
import sys
import urllib.request

import numpy as np

ROOT = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
TRAIN_BYTES = 64 << 20
DATA = pathlib.Path(__file__).resolve().parent / "data"
SEP = b"<|endoftext|>"

# sha256 of the processed .bin files, pinned on first preparation (2026-11-01)
EXPECTED = {
    "train.bin": "7392a20e6a56c6790e5da2e317bda54b62cc66c53a367354465d18222bbea12a",
    "heldout.bin": "ac3a314963fa0c1f5292b2ca7e442926d1ebd35eb66ca2e5902e705d558cbffc",
}


def fetch(name, limit=None):
    req = urllib.request.Request(ROOT + name, headers={"User-Agent": "odometer-prepare"})
    if limit:
        # a little extra so the cut can land on a story boundary
        req.add_header("Range", f"bytes=0-{limit + (1 << 16) - 1}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def to_bytes(raw, limit=None):
    stories = raw.split(SEP)
    if limit:
        out, n = [], 0
        for s in stories[:-1]:
            s = s.strip()
            if n + len(s) + 1 > limit:
                break
            out.append(s)
            n += len(s) + 1
        stories = out
    else:
        stories = [s.strip() for s in stories]
    blob = b"\0".join(s for s in stories if s) + b"\0"
    return np.frombuffer(blob, dtype=np.uint8)


def main():
    DATA.mkdir(exist_ok=True)
    jobs = [("train.bin", "TinyStoriesV2-GPT4-train.txt", TRAIN_BYTES), ("heldout.bin", "TinyStoriesV2-GPT4-valid.txt", None)]
    ok = True
    for out, src, limit in jobs:
        path = DATA / out
        if not path.exists():
            print(f"downloading {src} ...", flush=True)
            to_bytes(fetch(src, limit), limit).tofile(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        want = EXPECTED[out]
        status = "ok" if want in (None, digest) else "HASH MISMATCH"
        ok &= status == "ok"
        print(f"{out}: {path.stat().st_size:,} bytes  sha256 {digest[:16]}  {status}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
