#!/usr/bin/env python3
"""Disk hygiene for a Miles task's public suites, in the Work image only.

The Work image's `run_public.py` is a two-line wrapper that calls `run(<the task's public runner>, ...)`; the runner
itself (the same code as the Judge's checks) is unchanged. Around one suite this:

  - marks the suite's output directory when the suite ends (`.public_suite.json`: pid, start, end, pinned);
  - rotates: before the suite starts, the marked outputs next to it lose their bulky files (below), except the most
    recent finished one (KEEP_PREVIOUS) and the pinned ones. So the suite being run and the one before it stay
    complete, and the agent can inspect the last result without running it again.
  - --keep-artifacts pins the suite's output (never rotated; its space is the agent's to free) and passes the runner's
    own keep options (opd / opcd: --keep-snapshots, the checks' served-weight copies).

Bulky files are model weights, checkpoints, weight copies, step tensors, generated images and videos: every file of
BULKY_MIN_BYTES or more, and everything under a directory named in BULKY_DIRS (the contract scratch, checkpoints,
final weights, weight snapshots), except text records (TEXT_SUFFIXES: reports, logs, JSON), which always stay.
Only directories this module marked are pruned; nothing else under /scratch is touched.
"""
from __future__ import annotations

import json
import os
import runpy
import shutil
import signal
import sys
import time
from pathlib import Path

MARKER = ".public_suite.json"
KEEP_PREVIOUS = 1
BULKY_MIN_BYTES = 64 << 20
BULKY_DIRS = {"scratch", "tmp", "checkpoints", "ckpt", "final_hf", "snapshots", "artifacts"}
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".csv", ".md", ".yaml", ".yml", ".py", ".sh"}


def _is_text(p: Path) -> bool:
    return p.suffix.lower() in TEXT_SUFFIXES


def clean(out: Path) -> dict:
    """Remove the bulky files under `out` (keeping text records); return how much was freed."""
    freed, removed = 0, 0
    for root, dirs, files in os.walk(out, topdown=True):
        r = Path(root)
        in_bulky = any(part in BULKY_DIRS for part in r.relative_to(out).parts)
        for name in files:
            p = r / name
            if p.is_symlink() or _is_text(p) or name == MARKER:
                continue
            try:
                size = p.stat().st_size
                if in_bulky or size >= BULKY_MIN_BYTES:
                    p.unlink()
                    freed, removed = freed + size, removed + 1
            except OSError:
                pass
    for root, dirs, _ in os.walk(out, topdown=False):  # drop directories the removal emptied
        for d in dirs:
            try:
                (Path(root) / d).rmdir()
            except OSError:
                pass
    return {"files_removed": removed, "gib_freed": round(freed / 2**30, 2)}


def _read(marker: Path) -> dict:
    try:
        return json.loads(marker.read_text())
    except (OSError, ValueError):
        return {}


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def prune(parent: Path, current: Path) -> list[str]:
    """Clean the marked, unpinned outputs under `parent` beyond the KEEP_PREVIOUS most recent finished ones."""
    done = []
    for d in parent.iterdir() if parent.is_dir() else []:
        m = _read(d / MARKER)
        if not m or d.resolve() == current.resolve() or m.get("cleaned") or m.get("pinned"):
            continue
        if m.get("end") is None and _alive(m.get("pid")):
            continue  # another suite is still writing there
        done.append((m.get("start", 0), d, m))
    done.sort(key=lambda x: x[0], reverse=True)
    pruned = []
    for _, d, m in done[KEEP_PREVIOUS:]:
        m.update(cleaned=clean(d) | {"when": time.time(), "why": "rotated: a newer public suite started"})
        (d / MARKER).write_text(json.dumps(m))
        pruned.append(d.name)
    return pruned


def suite_dir(argv) -> str:
    """/scratch/public/<suite>-<time>: the output a runner without an output argument uses (the Wan2.2 runner)."""
    suite = "suite"
    for i, a in enumerate(argv):
        if a == "--suite" and i + 1 < len(argv):
            suite = argv[i + 1]
        elif a.startswith("--suite="):
            suite = a.split("=", 1)[1]
    return f"/scratch/public/{suite}-{time.strftime('%Y%m%dT%H%M%S')}"


def _sigterm(*_):
    raise SystemExit(143)


def run(runner: str, out_flag: str = "--output", default_out=None, keep_flags=()) -> None:
    """Run the public `runner` with this process's arguments, with the cleanup above. `out_flag` names the runner's
    output argument; `default_out(argv)` gives the directory the runner uses when the argument is absent (it is then
    passed explicitly); `keep_flags` are the runner's own keep options, added when --keep-artifacts is given."""
    argv = sys.argv[1:]
    keep = "--keep-artifacts" in argv
    argv = [a for a in argv if a != "--keep-artifacts"]
    out = None
    for i, a in enumerate(argv):
        if a == out_flag and i + 1 < len(argv):
            out = Path(argv[i + 1])
        elif a.startswith(out_flag + "="):
            out = Path(a.split("=", 1)[1])
    if out is None and default_out is not None and not any(a in ("-h", "--help") for a in argv):
        out = Path(default_out(argv))
        argv += [out_flag, str(out)]
    if keep:
        argv += [f for f in keep_flags if f not in argv]
    if out is None:  # --help or a usage error: nothing to manage
        sys.argv = [sys.argv[0], *argv]
        runpy.run_path(runner, run_name="__main__")
        return
    out = out.absolute()
    if out.exists():  # not a fresh suite output (the runners create theirs): leave it and its neighbours alone
        print(f"public_cleanup: {out} already exists, so it is not managed (no cleanup)", flush=True)
        sys.argv = [sys.argv[0], *argv]
        runpy.run_path(runner, run_name="__main__")
        return
    pruned = prune(out.parent, out)
    if pruned:
        print(f"public_cleanup: removed the bulky files of older public outputs {', '.join(pruned)}; their "
              "reports and logs stay (the previous output and this one stay complete)", flush=True)
    marker = {"pid": os.getpid(), "start": time.time(), "end": None, "pinned": keep, "argv": argv}
    signal.signal(signal.SIGTERM, _sigterm)
    sys.argv = [sys.argv[0], *argv]
    try:
        runpy.run_path(runner, run_name="__main__")
    finally:
        if out.is_dir():
            marker["end"] = time.time()
            (out / MARKER).write_text(json.dumps(marker))
            print(f"public_cleanup: {out} is kept complete " + ("(pinned by --keep-artifacts: delete it yourself when "
                  "done)" if keep else "until the second suite after it starts"), flush=True)
