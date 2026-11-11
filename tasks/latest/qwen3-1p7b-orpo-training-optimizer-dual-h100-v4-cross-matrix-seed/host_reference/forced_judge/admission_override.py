#!/usr/bin/env python3
"""Build-time patch for the LOCAL-ONLY forced-oracle Judge image.

Never used by the published Work or Judge image, and never part of a
participant submission.  Inside the derived image it wraps the per-file
source scan (``_check_python``) of the lab's canonical admission policy,
``runtime/_source_policy_impl.py``, so that any scan error (import allowlist,
attribute/dunder bans, size limits, ...) is recorded and then bypassed.  The
structural checks of ``admit()`` stay: contiguous ``objective_vNNN`` versions
holding exactly ``candidate.py`` + ``__init__.py``, a valid ``FINAL_VERSION``
and a ``notes.md``.  Everything after admission -- the hidden CPU objective
cases, the five-row formal E2E controller (paired official-TRL reference,
loss / auxiliary / gradient / AdamW / memory gates), aggregation and the
report -- is the unmodified Judge.

``protected_hashes.txt`` is updated so the verifier's own protected-file
self-check covers the patched file instead of failing on it.  The bypassed
errors are written (best effort) to ``/logs/verifier/forced_oracle_admission.json``.

Usage (Dockerfile):
    python3 admission_override.py --lab /app/workdir/<lab> \
        --manifest /opt/<judge>/protected_hashes.txt
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

MARKER = "# ---- FORCED-ORACLE ADMISSION OVERRIDE (local-only image) ----"
OVERRIDE = '''

# ---- FORCED-ORACLE ADMISSION OVERRIDE (local-only image) ----
_forced_oracle_original_check_python = _check_python
_forced_oracle_bypassed = []


def _check_python(path):
    try:
        return _forced_oracle_original_check_python(path)
    except AdmissionError as exc:
        _forced_oracle_bypassed.append({"file": str(path), "error": str(exc)})
        try:
            import json as _json

            Path("/logs/verifier/forced_oracle_admission.json").write_text(
                _json.dumps(
                    {"forced_oracle": True, "bypassed_errors": _forced_oracle_bypassed},
                    indent=2,
                    sort_keys=True,
                )
                + "\\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        return None
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    impl = args.lab / "runtime/_source_policy_impl.py"
    relative = impl.relative_to("/").as_posix()
    text = impl.read_text(encoding="utf-8")
    if MARKER in text:
        raise SystemExit("policy already patched")
    if "def _check_python(" not in text or "class AdmissionError" not in text:
        raise SystemExit("unexpected policy module layout")
    lines = args.manifest.read_text(encoding="utf-8").splitlines()
    before = sha256(impl)
    out, hits = [], 0
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[1] == relative:
            if parts[0] != before:
                raise SystemExit("policy file does not match the protected manifest")
            hits += 1
            out.append(None)
            continue
        out.append(line)
    if hits != 1:
        raise SystemExit("policy file is not in the protected manifest exactly once")
    impl.write_text(text + OVERRIDE, encoding="utf-8")
    after = sha256(impl)
    out = [f"{after}  {relative}" if line is None else line for line in out]
    args.manifest.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"patched {relative}: {before[:12]} -> {after[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
