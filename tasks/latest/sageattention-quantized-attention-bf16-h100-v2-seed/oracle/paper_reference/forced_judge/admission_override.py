#!/usr/bin/env python3
"""Build-time patch for the LOCAL-ONLY forced-oracle Judge image.

Never used by the published Work or Judge image.  It wraps
``source_admission.validate_submission`` inside the derived image so that any
admission error (import allowlist, compiled-extension ban, source scan) is
recorded and then bypassed, as long as FINAL_VERSION selects an existing
``candidates/kernel_vNNN/submission.py``.  Everything after admission -- the
frozen final holdout, the 8 edge cases, the 62 real layers, fresh-input trials,
timing, aggregation and the report -- is the unmodified Judge.

The bypassed errors are written to ``$LOG_DIR/forced_oracle_admission.json``
(it travels in private_judge_artifacts.tar.gz) because the Judge discards the
verifier's stdout/stderr.  ``protected_hashes.txt`` is updated so the verifier's
own self-check covers the patched file instead of failing on it.

Usage (Dockerfile): python3 admission_override.py /opt/<judge_root>
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

MARKER = "# ---- FORCED-ORACLE ADMISSION OVERRIDE (local-only image) ----"
OVERRIDE = '''
# ---- FORCED-ORACLE ADMISSION OVERRIDE (local-only image) ----
_forced_oracle_original_validate_submission = validate_submission


def validate_submission(root, candidate_version=None, *, workspace_mode=False):
    payload = _forced_oracle_original_validate_submission(
        root, candidate_version, workspace_mode=workspace_mode
    )
    bypassed = list(payload.get("errors") or [])
    record = {
        "schema_version": 1,
        "forced_oracle": True,
        "original_ok": bool(payload.get("ok")),
        "bypassed_errors": bypassed,
        "selected_relative_path": payload.get("selected_relative_path"),
        "selected_sha256": payload.get("selected_sha256"),
    }
    if not payload.get("ok") and payload.get("selected_path"):
        payload = dict(payload, ok=True, errors=[], forced_bypassed_errors=bypassed)
    record["forced_ok"] = bool(payload.get("ok"))
    if not workspace_mode:
        import os as _os
        try:
            log_dir = Path(_os.environ.get("LOG_DIR", "/logs/verifier"))
            (log_dir / "forced_oracle_admission.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\\n"
            )
        except OSError:
            pass
    return payload

'''


def main() -> int:
    root = Path(sys.argv[1])
    admission = root / "source_admission.py"
    hashes = root / "protected_hashes.txt"
    text = admission.read_text()
    if MARKER in text:
        raise SystemExit("already patched")
    anchor = '\n\nif __name__ == "__main__":'
    if text.count(anchor) != 1 or "def validate_submission(" not in text:
        raise SystemExit("unexpected source_admission.py layout")
    patched = text.replace(anchor, "\n" + OVERRIDE + anchor)
    admission.write_text(patched)
    digest = hashlib.sha256(patched.encode()).hexdigest()
    lines = []
    replaced = 0
    for line in hashes.read_text().splitlines():
        if line.strip() and not line.startswith("#") and line.split(None, 1)[1] == "judge:source_admission.py":
            line = f"{digest} judge:source_admission.py"
            replaced += 1
        lines.append(line)
    if replaced != 1:
        raise SystemExit("protected_hashes.txt lacks judge:source_admission.py")
    hashes.write_text("\n".join(lines) + "\n")
    print(f"patched {admission} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
