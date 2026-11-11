#!/usr/bin/env python3
"""FORCED-ORACLE admission override.  LOCAL ONLY; never part of a published image.

Installed by ``Dockerfile`` (this directory) as the Judge's ``source_admission.py``
inside a derived, host-local image tagged ``...:forced-oracle``.  The task's real
admission module is kept next to it, unmodified, as ``source_admission_enforcing.py``
and still runs: every structural decision (submission root layout, FINAL_VERSION,
candidate enumeration, the selected path and its SHA-256) is the enforcing module's.
Only the verdict is overridden: when a selected source exists, the decision is
``ok`` and the policy errors (import allowlist, compiled-extension and existing-
kernel bans, ...) are moved to ``warnings`` and ``forced_admission_bypassed_errors``.

Purpose: score the upstream paper implementation (which is not admissible to an
agent) through the otherwise unchanged Judge pipeline.  Never use this for an
agent archive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import source_admission_enforcing as _enforcing

FORCED_ADMISSION = True


def validate_submission(
    root: Path,
    candidate_version: int | None = None,
    *,
    workspace_mode: bool = False,
) -> dict[str, Any]:
    decision = _enforcing.validate_submission(
        root, candidate_version, workspace_mode=workspace_mode
    )
    selected = decision.get("selected_path")
    digest = decision.get("selected_sha256")
    final_version = decision.get("final_version")
    if (
        isinstance(selected, str)
        and Path(selected).is_file()
        and isinstance(digest, str)
        and isinstance(final_version, int)
        and not isinstance(final_version, bool)
        and final_version > 0
    ):
        bypassed = [str(error) for error in decision.get("errors", [])]
        decision["ok"] = True
        decision["errors"] = []
        decision["warnings"] = list(decision.get("warnings", [])) + [
            f"FORCED ADMISSION (local forced-oracle image) bypassed: {error}" for error in bypassed
        ]
        decision["forced_admission_bypassed_errors"] = bypassed
    decision["forced_admission"] = True
    return decision


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate-version", type=int)
    parser.add_argument("--workspace-mode", action="store_true")
    args = parser.parse_args(argv)
    payload = validate_submission(
        args.root, args.candidate_version, workspace_mode=args.workspace_mode
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
