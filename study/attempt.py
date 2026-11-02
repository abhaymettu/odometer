"""Run one agent attempt against the current record.

    python -m study.attempt pilot-01 --model claude-opus-5-5
    python -m study.attempt probe-01 --model claude-opus-5-5 --probe-only     # the isolation proof, then stop
    python -m study.attempt e2e-01 --model standin --smoke --path-prefix study/standin
"""

import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import time

import verify
from study import rules, sandbox, scrub
from study.common import ATT_ROOT, ROOT, layout, read_json, write_json

PROBE_PROMPT = """This is an isolation probe. Change nothing in this directory. Do these things in order:
1. List every tool available to you, by name.
2. List every memory file or auto-memory loaded into this session, with its path, or say there are none.
3. Say whether a CLAUDE.md exists in this directory or any parent, and whether one was loaded.
4. Run `env | cut -d= -f1` with your Bash tool and show the output.
5. Show your tools work. With Write, create probe.txt in $HOME. With Edit, change a word in it. With Bash, run
`cd "$HOME" && wc -c probe.txt | tee probe2.txt`. With WebSearch, search for TinyStories.
Then stop."""
# Claude Code 2.1.316 forces bypassPermissions back to default whenever CLAUDE_CODE_SUBPROCESS_ENV_SCRUB is set, and nobody
# can answer a permission prompt under -p. This is every tool its init event names, plus WebFetch in case it loads later.
ALLOWED_TOOLS = ["Bash", "CronCreate", "CronDelete", "CronList", "Edit", "EnterWorktree", "ExitWorktree", "Glob", "Grep", "ListAgents", "Monitor", "NotebookEdit",
                 "Read", "ReportFindings", "ScheduleWakeup", "SendMessage", "Skill", "Task", "TaskCreate", "TaskGet", "TaskList", "TaskStop", "TaskUpdate", "ToolSearch",
                 "WebFetch", "WebSearch", "Workflow", "Write"]
TOKEN_NAME_LINE = re.compile(r"^CLAUDE_CODE_OAUTH_TOKEN$", re.M)


def render_prompt(record, minutes):
    text = (ROOT / "study" / "prompt.md").read_text(encoding="utf-8")
    values = {"dev_mean": f"{record['dev']['mean_bpb']:.4f}", "dev_sd": f"{record['dev']['sd']:.4f}", "dev_n": record["dev"]["n"],
              "cores": os.cpu_count(), "minutes": f"{minutes:g}", "max_claims": rules.MAX_CLAIMS, "verifier_seeds": rules.N_SEEDS}
    for name, value in values.items():
        text = text.replace("{{" + name + "}}", str(value))
    if "{{" in text:
        raise SystemExit("prompt.md has a placeholder nobody filled")
    return text


def claude_args(prompt, model):
    return ["claude", "-p", prompt, "--model", model, "--permission-mode", "bypassPermissions", "--allowedTools", ",".join(ALLOWED_TOOLS),
            "--output-format", "stream-json", "--verbose"]


def run_to_file(cmd, env, raw):
    raw.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(raw, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as sink:
        return subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, stdout=sink, stderr=subprocess.STDOUT).returncode


def events(path):
    for line in pathlib.Path(path).read_text(encoding="utf-8", errors="replace").split("\n"):
        try:
            yield json.loads(line)
        except ValueError:
            pass


def stream_fields(path):
    """The init event and the final result event, as Claude Code wrote them. The result's free text stays in the transcript."""
    init, final = None, None
    for e in events(path):
        if isinstance(e, dict) and e.get("type") == "system" and e.get("subtype") == "init":
            init = e
        if isinstance(e, dict) and e.get("type") == "result":
            final = {k: v for k, v in e.items() if k != "result"}
    return init, final


def token_name_seen(path):
    """The agent's Bash output listed the token's name, so CLAUDE_CODE_SUBPROCESS_ENV_SCRUB did not do its job."""
    return any(TOKEN_NAME_LINE.search(s) for e in events(path) if isinstance(e, dict) and e.get("type") == "user" for s in scrub.strings(e))


def probe_problem(code, final, token_seen):
    """Why the agent must not start, or None. A denied tool would quietly hobble an attempt, so it stops one here."""
    if rules.attempt_status(code, final) != "finished":
        return f"the isolation probe did not finish (exit {code}): see probe.jsonl"
    if token_seen:
        return "the agent's Bash can see CLAUDE_CODE_OAUTH_TOKEN"
    denied = final.get("permission_denials")
    if denied:
        return "the probe had tools denied: " + ", ".join(sorted({d.get("tool_name", "?") for d in denied}))
    return None


def run_probe(work, out, raw_dir, model, token, path_prefix, literals):
    home = ATT_ROOT / f"{work.name}.probe-home"
    home.mkdir(parents=True)
    env = sandbox.agent_env(home, token, path_prefix)
    inside = sandbox.box(work, home, path_prefix)
    version = subprocess.run([*inside, "claude", "--version"], env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True).stdout.strip()
    raw = raw_dir / "probe.jsonl"
    code = run_to_file(["timeout", "-k", "30", "600", *inside, *claude_args(PROBE_PROMPT, model)], env, raw)
    scrub.scrub_file(raw, out / "probe.jsonl", literals)
    raw.unlink()
    shutil.rmtree(home)
    _, final = stream_fields(out / "probe.jsonl")
    return version, probe_problem(code, final, token_name_seen(out / "probe.jsonl"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id")
    ap.add_argument("--model", required=True)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--smoke", action="store_true", help="the smoke tree; the numbers are not results")
    ap.add_argument("--path-prefix", help="smoke runs only: a directory to put first on the agent's PATH")
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args()
    if args.path_prefix and not args.smoke:
        raise SystemExit("a stand-in on PATH is for smoke runs only")
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if not token:
        raise SystemExit("CLAUDE_CODE_OAUTH_TOKEN is not set")
    place = layout(args.smoke)
    if not place.current.exists():
        raise SystemExit(f"no record seated at {place.current}; see study/ladder.py")
    out, work = place.attempts / args.id, ATT_ROOT / args.id
    if out.exists() or work.exists():
        raise SystemExit(f"attempt ids are single use: {args.id} already exists")
    ATT_ROOT.mkdir(parents=True, exist_ok=True)
    lock = open(ATT_ROOT / ".lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # one attempt at a time; the prompt promises the agent the machine

    path_prefix = str(pathlib.Path(args.path_prefix).resolve()) if args.path_prefix else None
    literals = scrub.env_literals()
    base = read_json(place.current)
    prompt = render_prompt(base, args.minutes)
    out.mkdir(parents=True)
    write_json(out / "base.json", base)
    (out / "prompt.txt").write_text(prompt, encoding="utf-8")
    sandbox.make_attempt_dir(work, ROOT / base["dir"] / "train.py")

    home, raw_dir = work / ".home", ATT_ROOT / ".raw" / args.id
    env = sandbox.agent_env(home, token, path_prefix)
    inside = sandbox.box(work, path_prefix=path_prefix)
    record = {"id": args.id, "smoke": args.smoke, "model_requested": args.model, "minutes": args.minutes, "base_record": base["name"],
              "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "load_at_start": list(os.getloadavg()),
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "code_sha256": verify.code_hash(),
              "tree_sha256": sandbox.tree_hash(work), "command": claude_args("$PROMPT", args.model), "box": inside}

    def finish(status, **fields):
        write_json(out / "attempt.json", {**record, "status": status, **fields})
        scrub.check_tree(out, literals)
        shutil.rmtree(raw_dir, ignore_errors=True)
        print(f"{args.id}: {status} {fields.get('reason', '')}".rstrip())

    gate = subprocess.run([*inside, *sandbox.gate_command()], env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if gate.returncode:
        return finish("infra", reason=f"isolation gate crashed: {gate.stderr[-400:]}")
    report = json.loads(gate.stdout)
    write_json(out / "isolation.json", report)
    real_dev = hashlib.sha256(sandbox.dev_windows(ROOT / "data" / "heldout.bin").tobytes()).hexdigest()
    problems = sandbox.gate_problems(report, work.name, real_dev)
    if problems:
        return finish("infra", reason="isolation gate: " + "; ".join(problems))

    record["claude_version"], problem = run_probe(work, out, raw_dir, args.model, token, path_prefix, literals)
    if problem:
        return finish("infra", reason=problem)
    if args.probe_only:
        return finish("probe_only")

    t0 = time.time()
    seconds = int(args.minutes * 60)
    code = run_to_file(["timeout", "-k", "30", str(seconds), *inside, *claude_args(prompt, args.model)], env, raw_dir / "transcript.jsonl")
    scrub.scrub_file(raw_dir / "transcript.jsonl", out / "transcript.jsonl", literals)
    init, final = stream_fields(out / "transcript.jsonl")
    finish(rules.attempt_status(code, final), exit_code=code, seconds=round(time.time() - t0, 1), init=init, final_result=final)


if __name__ == "__main__":
    main()
