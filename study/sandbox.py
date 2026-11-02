"""What an agent gets: a copy of the track with the test split zeroed, and a view of this machine with nothing else in it."""

import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile

import verify
from study.common import ROOT

CONTEXT = verify.TRACK["eval_context"]
ENV_NAMES = ["CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB", "HOME", "LANG", "PATH", "PWD"]  # bwrap --chdir always sets PWD
STANDIN_MOUNT = "/opt/standin"  # a mount under the repo would make the repo's study/ directory exist inside the box
# An allowlist: design notes, review/ and this folder state the study's goals and list exploits, and a new note
# in the repo must not reach an agent by default.
COPIED = ["verify.py", "stats.py", "odometer/*.py", "entries/*/train.py", "tools/calibrate.py", "tests/conftest.py", "tests/test_engine.py",
          "tests/test_attacks.py", "tests/attacks/*"]
ETC = ["resolv.conf", "hosts", "nsswitch.conf", "passwd", "group", "ssl", "ca-certificates", "alternatives", "ld.so.cache", "localtime", "gai.conf"]


def dev_windows(heldout, context=CONTEXT):
    from odometer import _data

    saved, _data.DATA = _data.DATA, pathlib.Path(heldout).parent
    try:
        return _data.heldout_windows("dev", context, 1 << 30)
    finally:
        _data.DATA = saved


def seal(src, dst, context=CONTEXT):
    """Copy heldout.bin keeping only the dev windows; every other byte becomes zero and the size stays the same."""
    import numpy as np

    width = context + 1
    real = np.fromfile(src, np.uint8)
    n = real.size // width
    # Ask the track's own split which windows are dev, through a throwaway file whose windows spell their index.
    ids = np.arange(n)
    index = np.zeros((n, width), np.uint8)
    index[:, 0], index[:, 1], index[:, 2] = ids >> 16, (ids >> 8) & 255, ids & 255
    with tempfile.TemporaryDirectory() as tmp:
        index.tofile(pathlib.Path(tmp) / "heldout.bin")
        rows = dev_windows(pathlib.Path(tmp) / "heldout.bin", context).astype(np.int64)
    dev = rows[:, 0] << 16 | rows[:, 1] << 8 | rows[:, 2]
    sealed = np.zeros((n, width), np.uint8)
    sealed[dev] = real[: n * width].reshape(n, width)[dev]
    sealed = np.concatenate([sealed.ravel(), np.zeros(real.size - n * width, np.uint8)])
    sealed.tofile(dst)


def make_attempt_dir(work, record_entry, repo=ROOT):
    work = pathlib.Path(work)
    for pattern in COPIED:
        for src in repo.glob(pattern):
            rel = src.relative_to(repo)
            if src.is_file():
                (work / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, work / rel)
    for d in (".home", ".tmp", "claims", "runs", "data", "entries/current"):
        (work / d).mkdir(parents=True, exist_ok=True)
    shutil.copy2(record_entry, work / "entries" / "current" / "train.py")
    shutil.copy2(repo / "data" / "train.bin", work / "data" / "train.bin")
    seal(repo / "data" / "heldout.bin", work / "data" / "heldout.bin")


def tree_hash(work):
    h = hashlib.sha256()
    for p in sorted(q for q in pathlib.Path(work).rglob("*") if q.is_file()):
        h.update(str(p.relative_to(work)).encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def agent_env(home, token, path_prefix=None):
    """What `env -i` would pass. A dict rather than an env -i command line, so the token is never in anyone's argv."""
    path = [STANDIN_MOUNT] if path_prefix else []
    return {"PATH": ":".join([*path, f"{sys.prefix}/bin", "/usr/local/bin", "/usr/bin", "/bin"]), "HOME": str(home),
            "LANG": "C.UTF-8", "CLAUDE_CODE_OAUTH_TOKEN": token, "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB": "1"}


def box(work, home=None, path_prefix=None):
    """bwrap argv. A fresh HOME and env do not stop an absolute path: without this the agent could read the real test
    split, earlier attempts, the VM's login, and other processes' environments through /proc. With home given, the
    attempt directory is read-only and that home is the only writable place (the isolation probe uses this). setpriv
    drops ambient capabilities first: boat exec hands out CAP_DAC_READ_SEARCH, and bwrap refuses to start with it."""
    args = ["setpriv", "--ambient-caps=-all", "--inh-caps=-all", "bwrap", "--die-with-parent", "--new-session", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/sbin", "/sbin",
            "--ro-bind", "/sys", "/sys", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/dev/shm",
            "--tmpfs", "/home", "--tmpfs", "/var", "--tmpfs", "/run", "--tmpfs", "/tmp"]
    for name in ETC:
        args += ["--ro-bind-try", f"/etc/{name}", f"/etc/{name}"]
    args += ["--ro-bind", sys.prefix, sys.prefix]
    args += ["--ro-bind", str(path_prefix), STANDIN_MOUNT] if path_prefix else []
    if home:
        args += ["--ro-bind", str(work), str(work), "--bind", str(home), str(home)]
    else:
        args += ["--bind", str(work), str(work), "--bind", str(pathlib.Path(work) / ".tmp"), "/tmp"]
    return args + ["--chdir", str(work)]


GATE_SCRIPT = """
import hashlib, json, os, pathlib, sys
import numpy as np
from odometer import _data
cwd = pathlib.Path.cwd()
test = _data.heldout_windows("test", int(sys.argv[2]), 1 << 30)
dev = _data.heldout_windows("dev", int(sys.argv[2]), 1 << 30)
print(json.dumps({
    "env": sorted(os.environ),
    "home_entries": sorted(os.listdir(os.environ["HOME"])),
    "visible": [p for p in json.loads(sys.argv[1]) if os.path.lexists(p)],
    "siblings": sorted(os.listdir(cwd.parent)),
    "pids": sum(p.isdigit() for p in os.listdir("/proc")),
    "capabilities": [l.split()[1] for l in open("/proc/self/status") if l.startswith("CapEff")][0],
    "claude_config": [str(p / n) for p in [cwd, *cwd.parents] for n in ("CLAUDE.md", ".claude") if (p / n).exists()],
    "test_windows": len(test), "test_nonzero_bytes": int(np.count_nonzero(test)),
    "dev_windows": len(dev), "dev_sha256": hashlib.sha256(dev.tobytes()).hexdigest(),
}))
"""


def hidden_paths():
    home = pathlib.Path.home()
    return [str(home / ".claude"), str(home / ".claude.json"), str(ROOT / "data"), str(ROOT / "verify.py"), str(ROOT / "study"),
            str(ROOT / "records"), str(ROOT / "runs"), "/etc/claude-code", "/var/run/docker.sock", "/run/docker.sock"]


def gate_command():
    return ["python", "-c", GATE_SCRIPT, json.dumps(hidden_paths()), str(CONTEXT)]


def gate_problems(report, work_name, real_dev_sha):
    """Checked before any model call. The report comes from inside the box, under the agent's own env."""
    checks = {
        "the environment has names outside the allowed list": report["env"] != ENV_NAMES,
        "HOME is not empty": report["home_entries"] != [],
        f"visible from inside: {report['visible']}": report["visible"] != [],
        f"other attempts are visible: {report['siblings']}": report["siblings"] != [work_name],
        f"processes outside the box are visible ({report['pids']} pids)": report["pids"] > 3,
        f"the agent holds capabilities: {report['capabilities']}": int(report["capabilities"], 16) != 0,
        f"Claude config reachable: {report['claude_config']}": report["claude_config"] != [],
        "the test split has nonzero bytes": report["test_nonzero_bytes"] != 0 or report["test_windows"] == 0,
        "the dev split differs from the real one": report["dev_sha256"] != real_dev_sha,
    }
    return [problem for problem, failed in checks.items() if failed]
