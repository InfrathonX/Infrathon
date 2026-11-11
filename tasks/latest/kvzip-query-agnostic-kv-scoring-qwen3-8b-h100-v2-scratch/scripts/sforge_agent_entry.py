#!/usr/bin/env python3
"""Host agent-run entry: the task adapter plus two run-scoped bindings.

``scripts/run_agent.sh`` drives ``serve`` and ``run`` through this file.  It
imports ``sforge_entry`` unchanged (read-only Qwen3 mount, multi-GPU Judge
device request from ``KV_SCORING_JUDGE_GPU_UUIDS``, internal Work network,
task-JSON internet for Codex), then adds:

* the interactive Judge timeout for the Judge GPU count of this run.  The task
  JSON's ``judge.eval_timeout`` (11,000 s) covers three or more Judge GPUs;
  with one or two the suite takes longer, so the launcher passes
  ``scripts/lib.sh`` ``judge_timeouts``' value in ``KVZIP_JUDGE_EVAL_TIMEOUT``
  and both the Judge server and the Work-side ``sforge-submit`` wait use it.
  It may only raise the task value;
* Claude Code, when the launcher selected it (``MLSYSFORGE_AGENT=claude-code``;
  ``sforge/harness/agent/claude_local.py``), with the adapter's own internet rule.

The adapter file itself stays untouched: it is part of the task's
build-resource manifest (scripts/write_portable_manifests.py).
"""

from __future__ import annotations

import os
import sys

import sforge_entry as task_adapter
from sforge import cli as sforge_cli
from sforge.harness import task_spec as task_spec_module
from sforge.harness.agent.claude_local import install_from_environment

EVAL_TIMEOUT_ENV = "KVZIP_JUDGE_EVAL_TIMEOUT"
TASK_ID = task_adapter.TASK_ID
# The adapter's rule: MLSYSFORGE_WORK_INTERNET (the task JSON's internet value).
_task_internet = task_adapter._task_internet
main = task_adapter.main


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


if __name__ == "__main__":
    _install_eval_timeout(_eval_timeout_override())
    install_from_environment(_task_internet, sys.argv)
    main()
