"""Keep credentials out of everything the study keeps.

    python -m study.scrub study/attempts      # check a tree before publishing; exits 1 and names files if anything is left

Scrubbing rewrites what it recognizes. The separate check then looks again, on decoded JSON and for encoded copies of
the environment's secrets, and refuses to keep the file if anything is left. Neither ever prints a matched value.
"""

import base64
import json
import math
import os
import pathlib
import re
import sys
import urllib.parse
from collections import Counter

SECRET_NAME = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL", re.I)
SHAPES = {
    "anthropic-key": re.compile(r"sk-ant[\w-]*"),
    "github-token": re.compile(r"\b(?:gh[oprsu]_|github_pat_)\w*"),
    "bearer": re.compile(r"\bbearer\s+[\w.~+/=-]{8,}", re.I),
    "oauth-value": re.compile(r"oauth[\w-]*[\"']?\s*[:=]\s*[\"']?[\w.~+/=-]{8,}", re.I),
    "private-path": re.compile(r"/Users/[^/\s\"']+"),
}
LONG_RUN = re.compile(r"[A-Za-z0-9+/_-]{40,}={0,2}")


class Leak(Exception):
    pass


def env_literals(env=os.environ):
    return sorted({v for k, v in env.items() if SECRET_NAME.search(k) and len(v) >= 8}, key=len, reverse=True)


def looks_random(run):
    """Base64 credentials, not digests or paths: mixed case with digits, and about as random as base64 gets."""
    body = run.rstrip("=")
    if not (re.search(r"[a-z]", body) and re.search(r"[A-Z]", body) and re.search(r"[0-9]", body)):
        return False
    counts = Counter(body).values()
    return -sum(c / len(body) * math.log2(c / len(body)) for c in counts) >= 4.5


def scrub_string(s, literals):
    for value in literals:
        s = s.replace(value, "[scrubbed:literal]")
    for name, shape in SHAPES.items():
        s = shape.sub(f"[scrubbed:{name}]", s)
    return LONG_RUN.sub(lambda m: "[scrubbed:long-base64]" if looks_random(m[0]) else m[0], s)


def scrub_value(v, literals):
    if isinstance(v, str):
        return scrub_string(v, literals)
    if isinstance(v, list):
        return [scrub_value(x, literals) for x in v]
    if isinstance(v, dict):
        return {scrub_string(k, literals): scrub_value(x, literals) for k, x in v.items()}
    return v


def scrub_line(line, literals):
    """JSON is scrubbed after decoding, so an escape such as sk\\u002dant cannot hide a shape."""
    try:
        return json.dumps(scrub_value(json.loads(line), literals), ensure_ascii=False)
    except ValueError:
        return scrub_string(line, literals)


def strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, list):
        for x in v:
            yield from strings(x)
    elif isinstance(v, dict):
        for k, x in v.items():
            yield k
            yield from strings(x)


def encoded_forms(value):
    raw = value.encode()
    forms = {"literal": value, "base64": base64.b64encode(raw).decode().rstrip("="), "urlsafe-base64": base64.urlsafe_b64encode(raw).decode().rstrip("="),
             "hex": raw.hex(), "url-quoted": urllib.parse.quote(value, safe=""), "reversed": value[::-1]}
    return forms.items()


def leftovers(text, literals):
    """What the scrub left behind, by name only."""
    views = [text]
    for line in text.split("\n"):
        try:
            views.extend(strings(json.loads(line)))
        except ValueError:
            pass
    found = set()
    for view in views:
        found |= {name for name, shape in SHAPES.items() if shape.search(view)}
        if any(looks_random(m[0]) for m in LONG_RUN.finditer(view)):
            found.add("long-base64")
        found |= {f"secret-as-{form}" for value in literals for form, enc in encoded_forms(value) if enc in view}
    return sorted(found)


def scrub_text(text, literals):
    """Split on newlines only. str.splitlines also breaks on U+2028, which is legal inside a JSON string."""
    clean = "\n".join(scrub_line(line, literals) for line in text.rstrip("\n").split("\n")) + "\n"
    left = leftovers(clean, literals)
    if left:
        raise Leak(f"still present after scrubbing: {', '.join(left)}")
    return clean


def scrub_file(src, dst, literals):
    """dst is only written when clean. On a Leak the raw file stays put, private, for a human."""
    clean = scrub_text(pathlib.Path(src).read_text(encoding="utf-8", errors="replace"), literals)
    pathlib.Path(dst).write_text(clean, encoding="utf-8")


def check_tree(root, literals):
    """Every file under root, including code that must stay byte-exact and so is checked, never rewritten."""
    bad = {}
    for path in sorted(p for p in pathlib.Path(root).rglob("*") if p.is_file()):
        left = leftovers(path.read_text(encoding="utf-8", errors="replace"), literals)
        if left:
            bad[str(path)] = left
    if bad:
        raise SystemExit("LEAK, do not publish: " + json.dumps(bad, indent=1))


if __name__ == "__main__":
    check_tree(sys.argv[1], env_literals())
    print(f"{sys.argv[1]}: clean")
