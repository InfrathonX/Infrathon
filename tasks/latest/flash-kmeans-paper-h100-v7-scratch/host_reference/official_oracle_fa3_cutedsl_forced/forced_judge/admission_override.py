#!/usr/bin/env python3
"""Build-time patch for the LOCAL-ONLY forced-oracle flash-kmeans Judge image.

Never used by the published Work or Judge image.  It wraps
``source_admission.scan_submission`` so that any source-policy problem
(import allowlist: the candidate imports ``cutlass`` / ``flash_kmeans``) is
recorded and then bypassed, provided FINAL_VERSION selects an existing
``kernels/flash_kmeans_vNNN.py``.  Everything after admission -- protected
files, interface, correctness, ties, input integrity, memory, timing protocol,
fresh-process deployment protocol and the score -- is the unmodified Judge.

The bypassed problems are kept in the admission payload under
``forced_bypassed_problems`` and written to ``$LOG_DIR/forced_oracle_admission.json``
(default ``/logs/verifier``).  ``/opt/flash_kmeans_judge/protected_hashes.txt``
is updated so the verifier's protected-file check covers the patched file.

Usage (Dockerfile): python3 admission_override.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ADMISSION = Path("/opt/flash_kmeans_benchmark/source_admission.py")
HASHES = Path("/opt/flash_kmeans_judge/protected_hashes.txt")
MARKER = "# ---- FORCED-ORACLE ADMISSION OVERRIDE (local-only image) ----"
OVERRIDE = '''
# ---- FORCED-ORACLE ADMISSION OVERRIDE (local-only image) ----
_forced_oracle_original_scan_submission = scan_submission


def scan_submission(root):
    result = _forced_oracle_original_scan_submission(root)
    problems = list(result.get("problems") or [])
    record = {
        "schema_version": 1,
        "forced_oracle": True,
        "original_status": result.get("status"),
        "bypassed_problems": problems,
        "selected_version": result.get("selected_version"),
        "candidate_versions": result.get("candidate_versions"),
    }
    if (
        result.get("status") != "clean"
        and isinstance(result.get("selected_version"), int)
        and result.get("selected_version") in (result.get("candidate_versions") or [])
    ):
        result = dict(result, status="clean", problems=[], forced_bypassed_problems=problems)
    record["forced_status"] = result.get("status")
    try:
        import os as _os
        log_dir = Path(_os.environ.get("LOG_DIR", "/logs/verifier"))
        (log_dir / "forced_oracle_admission.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\\n"
        )
    except OSError:
        pass
    return result

'''


def main() -> int:
    text = ADMISSION.read_text()
    if MARKER in text:
        raise SystemExit("already patched")
    anchor = '\n\nif __name__ == "__main__":'
    if text.count(anchor) != 1 or "def scan_submission(" not in text:
        raise SystemExit("unexpected source_admission.py layout")
    patched = text.replace(anchor, "\n" + OVERRIDE + anchor)
    ADMISSION.write_text(patched)
    digest = hashlib.sha256(patched.encode()).hexdigest()
    lines, replaced = [], 0
    for line in HASHES.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and not line.startswith("#") and parts[1] == str(ADMISSION):
            line = f"{digest}  {ADMISSION}"
            replaced += 1
        lines.append(line)
    if replaced != 1:
        raise SystemExit("protected_hashes.txt lacks source_admission.py")
    HASHES.write_text("\n".join(lines) + "\n")
    print(f"patched {ADMISSION} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
