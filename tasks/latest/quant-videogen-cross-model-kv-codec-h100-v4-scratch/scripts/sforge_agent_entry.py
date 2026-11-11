#!/usr/bin/env python3
"""Host agent-run entry: the task adapter plus two run-scoped bindings.

``scripts/run_agent.sh`` drives ``serve`` and ``run`` through this file.  It
imports ``sforge_runtime_entry`` unchanged (GPU pinning, read-only model
mounts, Codex web search), then adds:

* the interactive Judge timeout for the Judge GPU count of this run.  The task
  JSON's ``judge.eval_timeout`` is sized for the full eight-GPU Judge; with
  fewer GPUs the controller's own list-scheduling bound is longer, so the
  launcher passes ``scripts/lib.sh`` ``judge_plan``'s value in
  ``QVG_JUDGE_EVAL_TIMEOUT`` and both the Judge server and the Work-side
  ``sforge-submit`` wait use it.  It may only raise the task value;
* Claude Code, when the launcher selected it (``MLSYSFORGE_AGENT=claude-code``;
  ``sforge/harness/agent/claude_local.py``).  The task allows internet, so the
  Agent keeps WebSearch/WebFetch.

The adapter file itself stays untouched: it is part of the task's
build-resource manifest.
"""

from __future__ import annotations

import os
import sys

import sforge_runtime_entry as task_adapter
from sforge import cli as sforge_cli
from sforge.harness import task_spec as task_spec_module
from sforge.harness.agent.claude_local import install_from_environment

EVAL_TIMEOUT_ENV = "QVG_JUDGE_EVAL_TIMEOUT"
TASK_ID = "quant_videogen_cross_model_kv_codec_h100_scratch_v4"


def _eval_timeout_override() -> int | None:
    raw = os.environ.get(EVAL_TIMEOUT_ENV, "").strip()
    if not raw:
        return None
    if not raw.isdigit() or int(raw) <= 0:
        raise SystemExit(f"{EVAL_TIMEOUT_ENV} must be a positive integer")
    return int(raw)


def _install_eval_timeout(timeout: int | None) -> None:
    if timeout is None:
        return
    original = task_spec_module.make_task_spec

    def make_task_spec(*args, **kwargs):
        spec = original(*args, **kwargs)
        if spec.task_id == TASK_ID:
            if timeout < spec.judge.eval_timeout:
                raise SystemExit(
                    f"{EVAL_TIMEOUT_ENV}={timeout} is below the task's "
                    f"judge.eval_timeout {spec.judge.eval_timeout}"
                )
            spec.judge.eval_timeout = timeout
        return spec

    # load_all_tasks resolves make_task_spec in its own module; the CLI
    # imported the name directly.
    task_spec_module.make_task_spec = make_task_spec
    sforge_cli.make_task_spec = make_task_spec


def _task_internet(requested: bool) -> bool:
    del requested
    return True  # task JSON: "internet": true


if __name__ == "__main__":
    _install_eval_timeout(_eval_timeout_override())
    install_from_environment(_task_internet, sys.argv)
    task_adapter.main()
