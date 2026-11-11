#!/usr/bin/env python3
from __future__ import annotations
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    ("judge:candidate_worker.py", ROOT / "images/judge/candidate_worker.py"),
    ("judge:trusted_replay.py", ROOT / "images/judge/trusted_replay.py"),
    ("judge:judge_shards.py", ROOT / "images/judge/judge_shards.py"),
    ("judge:released_weights_gate.py", ROOT / "images/judge/released_weights_gate.py"),
    ("judge:run_judge.py", ROOT / "images/judge/run_judge.py"),
    ("judge:source_admission.py", ROOT / "images/judge/source_admission.py"),
    ("judge:verify_kv_scoring.py", ROOT / "images/judge/verify_kv_scoring.py"),
    ("worker:kv_scoring_runtime.py", ROOT / "runtime/kv_scoring_runtime.py"),
    ("worker:kv_scoring_metric.py", ROOT / "runtime/kv_scoring_metric.py"),
)
def main() -> None:
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()} {label}" for label, path in FILES]
    (ROOT / "images/judge/protected_hashes.txt").write_text("\n".join(lines) + "\n")
if __name__ == "__main__": main()
