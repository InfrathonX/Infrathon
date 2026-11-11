"""Forced-oracle admission override (LOCAL-ONLY Judge image, never for agents).

Runs the task's enforced admission unchanged, keeps its full verdict under
``enforced_admission`` and its errors under ``forced_bypass_errors``, then
admits the candidate. Every later gate (worker protocol, teacher-forced oracle,
termination, determinism, wall/memory caps, idleness) is left unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from source_admission_enforced import *  # noqa: F401,F403  (keep helper names importable)
from source_admission_enforced import validate_submission as _enforced_validate_submission


def validate_submission(
    root: str | Path,
    candidate_version: int | None = None,
    *,
    workspace_mode: bool = False,
) -> dict[str, Any]:
    """Return the enforced verdict, forced to ``ok`` for the oracle archive."""
    verdict = dict(
        _enforced_validate_submission(
            root, candidate_version, workspace_mode=workspace_mode
        )
    )
    errors = list(verdict.get("errors") or [])
    verdict["enforced_admission"] = {"ok": bool(verdict.get("ok")), "errors": errors}
    verdict["forced_bypass_errors"] = errors
    verdict["forced_oracle"] = True
    if verdict.get("candidate_version") is None:
        verdict["candidate_version"] = candidate_version
    verdict["ok"] = True
    verdict["errors"] = []
    return verdict
