"""Forced-oracle source admission (LOCAL forced-oracle Judge image only).

Runs the Judge's own ``source_admission.validate_submission`` unchanged and
keeps its decision.  When that decision rejects the submission (for the
paper oracle: ``import deep_gemm``, a compiled extension), the submission
is admitted anyway and the original errors are recorded next to the forced
decision, so the private result says exactly what was bypassed.

Structural requirements stay: FINAL_VERSION must name an existing, regular,
non-empty ``candidates/kernel_vNNN/submission.py``.  Nothing here is copied
into the task's Work or Judge image; it exists only in the derived
``<judge image>:forced-oracle`` tag built by ``run_forced_oracle.sh``.
"""

from __future__ import annotations

import hashlib
import re
import stat
from pathlib import Path
from typing import Any

from source_admission import validate_submission as _original_validate_submission

_CANDIDATE_RE = re.compile(r"^kernel_v([0-9]{3})$")


def _versions(root: Path) -> list[int]:
    found: list[int] = []
    candidates = root / "candidates"
    try:
        entries = sorted(candidates.iterdir())
    except OSError:
        return found
    for entry in entries:
        match = _CANDIDATE_RE.fullmatch(entry.name)
        if match is not None and entry.is_dir() and not entry.is_symlink():
            found.append(int(match.group(1)))
    return sorted(found)


def validate_submission(
    root: Path, candidate_version: int | None = None, **kwargs: Any
) -> dict[str, Any]:
    root = Path(root)
    decision = _original_validate_submission(root, candidate_version, **kwargs)
    if decision.get("ok"):
        return dict(decision) | {"forced_bypass": False}
    original = {
        "ok": bool(decision.get("ok")),
        "errors": list(decision.get("errors") or []),
        "warnings": list(decision.get("warnings") or []),
    }
    try:
        text = (root / "FINAL_VERSION").read_text(encoding="utf-8").strip()
        final_version = int(text)
    except (OSError, UnicodeError, ValueError):
        return dict(decision) | {"forced_bypass": False, "forced_error": "FINAL_VERSION unreadable"}
    version = final_version if candidate_version is None else int(candidate_version)
    relative = f"candidates/kernel_v{version:03d}/submission.py"
    path = root / relative
    try:
        metadata = path.lstat()
        payload = path.read_bytes()
    except OSError:
        return dict(decision) | {"forced_bypass": False, "forced_error": f"{relative} unreadable"}
    if not stat.S_ISREG(metadata.st_mode) or not payload:
        return dict(decision) | {"forced_bypass": False, "forced_error": f"{relative} is not a non-empty regular file"}
    versions = _versions(root)
    return {
        "ok": True,
        "final_version": final_version,
        "candidate_version": version,
        "num_candidate_versions": len(versions),
        "versions": versions,
        "selected_path": str(path),
        "selected_relative_path": relative,
        "selected_sha256": hashlib.sha256(payload).hexdigest(),
        "errors": [],
        "warnings": original["warnings"],
        "forced_bypass": True,
        "original_decision": original,
    }
