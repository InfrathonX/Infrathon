#!/usr/bin/env python3
"""Write the v2 case manifests and every file that binds the formal suite's hash.

Outputs (all derived from runtime/cases.py, so they cannot drift apart):
  runtime/manifests/public_cases.json   public suite, Work-visible
  build/private/final_cases.json        formal suite, Judge-only, mode 0600
  runtime/evaluation_protocol.json      hash-protected protocol read by the verifier
  portable/formal-suite-manifest.json   identity + policy, no private case parameters
  portable/selection-protocol.json      round routing and candidate selection
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "runtime"))
import cases  # noqa: E402

PROTOCOL_ID = "sdv2-pipeline-v2-full-final"
TIMEOUTS = {"judge_controller_seconds": 4200, "outer_judge_seconds": 4200, "worker_pass_seconds": 1800}
REFERENCE = "pristine runtime_v001 re-measured on the same devices in every evaluation"
SCORE = {
    "direction": "maximize",
    "formula": "geomean(steady_fps * (in_run_seed_p99 / candidate_p99) ** 0.25)",
    "name": "latency_weighted_fps",
    "unit": "fps",
    "steady_fps": "frames(chunks 12..end) / (emit(last chunk) - controller push(chunk 12))",
    "latency": "emit(chunk) - controller push(chunk), chunks 12..end",
}


def _write(path: Path, payload: dict, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if mode is not None:
        path.chmod(mode)


def main() -> int:
    cases.write_manifests(TASK / "runtime" / "manifests", TASK / "build" / "private")
    final_path = TASK / "build" / "private" / "final_cases.json"
    digest = hashlib.sha256(final_path.read_bytes()).hexdigest()
    final = json.loads(final_path.read_text(encoding="utf-8"))
    count = len(final["cases"])
    suite_id = final["suite"]
    _write(TASK / "runtime" / "evaluation_protocol.json", {
        "candidate_selection": {
            "final_and_host_replay": "FINAL_VERSION or a validated host candidate_version override",
            "interactive": "FINAL_VERSION; candidate_version override forbidden",
        },
        "case_count": count,
        "protocol_id": PROTOCOL_ID,
        "public_diagnostics": "Public inputs are diagnostic only; they never select a shorter formal scoring suite.",
        "reference": REFERENCE,
        "schema_version": 1,
        "score": SCORE,
        "selected_manifest_sha256": digest,
        "selection": f"Every admitted formal round runs all {count} frozen private cases. Round identity changes lifecycle and delivery only.",
        "suite_id": suite_id,
        "timeouts": TIMEOUTS,
    })
    _write(TASK / "portable" / "formal-suite-manifest.json", {
        "case_count": count,
        "evaluation_protocol_id": PROTOCOL_ID,
        "input_visibility": "Judge only; this manifest contains identity and policy, not private case parameters or prompts",
        "reference": REFERENCE,
        "role": "formal",
        "schema_version": 1,
        "score": SCORE,
        "selected_manifest_sha256": digest,
        "suite_id": suite_id,
        "timeouts": TIMEOUTS,
    })
    _write(TASK / "portable" / "selection-protocol.json", {
        "final": {"candidate_selection": "FINAL_VERSION or validated host replay override", "case_count": count,
                  "selected_manifest_sha256": digest, "suite_id": suite_id},
        "interactive": {"candidate_selection": "FINAL_VERSION without candidate_version override", "case_count": count,
                        "selected_manifest_sha256": digest, "suite_id": suite_id},
        "protocol_id": PROTOCOL_ID,
        "public": {"scored": False, "semantics": "Participant diagnostics remain separate from formal submission and terminal evaluation."},
        "reference": REFERENCE,
        "rounds": {"host": "final / manual-* / candidate-*", "interactive": "agent-N / auto-N, N is a positive ASCII integer"},
        "schema_version": 1,
        "score": SCORE,
        "timeouts": TIMEOUTS,
    })
    print(f"{suite_id}: {count} cases, sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
