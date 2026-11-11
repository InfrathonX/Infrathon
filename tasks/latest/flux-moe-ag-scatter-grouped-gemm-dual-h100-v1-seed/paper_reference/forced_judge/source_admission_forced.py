"""Forced-oracle admission override (LOCAL-ONLY Judge image, never for agents).

Runs the task's enforced admission unchanged, keeps its errors under
``forced_bypass_errors`` and admits the selected source. Every runtime gate
(edge fixtures, correctness, immutability, fresh input/output, rank agreement,
timing) is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from source_admission_enforced import *  # noqa: F401,F403
from source_admission_enforced import validate_submission as _enforced_validate_submission


def validate_submission(
    root: Path,
    candidate_version: int | None = None,
    *,
    workspace_mode: bool = False,
) -> dict[str, Any]:
    verdict = dict(
        _enforced_validate_submission(
            root, candidate_version, workspace_mode=workspace_mode
        )
    )
    errors = list(verdict.get("errors") or [])
    verdict["enforced_admission"] = {"ok": bool(verdict.get("ok")), "errors": errors}
    verdict["forced_bypass_errors"] = errors
    verdict["forced_oracle"] = True
    verdict["ok"] = verdict.get("selected_path") is not None
    verdict["errors"] = [] if verdict["ok"] else errors
    return verdict
