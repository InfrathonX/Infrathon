#!/usr/bin/env python3
"""Sanitizing structured-result adapter for the SDV2 pipeline verifier."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

VERIFIER = Path("/opt/sdv2_judge/verify_sdv2_pipeline.py")
# Two measured passes per evaluation: the in-run seed reference and the
# candidate.  The image pins the same number through SDV2_TOTAL_TIMEOUT.
CONTROLLER_TIMEOUT = int(os.environ.get("SDV2_TOTAL_TIMEOUT", "4200"))
GATES = (
    "profile_ok", "hashes_ok", "admission_ok", "seed_reference_ok", "worker_ok",
    "completeness_ok", "parity_ok", "liveness_ok", "resources_ok", "ttff_ok", "p99_ok",
    "construction_ok",
)
START_MARKER = ">>>>> Start Structured Result"
END_MARKER = ">>>>> End Structured Result"


def _run(app_dir: str, log_dir: str) -> int:
    process = subprocess.Popen(
        [sys.executable, str(VERIFIER), "--app-dir", app_dir, "--log-dir", log_dir],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        return process.wait(timeout=CONTROLLER_TIMEOUT)
    except subprocess.TimeoutExpired:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try: os.killpg(process.pid, sig)
            except ProcessLookupError: break
            try: process.wait(timeout=10); break
            except subprocess.TimeoutExpired: continue
        return -1


def _json(path: Path, limit: int) -> dict | None:
    try:
        if path.stat().st_size > limit: return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _derive_metrics(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("missing per-case rows")
    def positive(value):
        if isinstance(value, bool): raise ValueError("boolean metric")
        value = float(value)
        if not math.isfinite(value) or value <= 0: raise ValueError("invalid metric")
        return value
    def gm(values):
        return math.exp(sum(math.log(value) for value in values) / len(values))
    fps = [positive(row["steady_fps"]) for row in rows]
    ratios = [positive(row["push_to_emit_p99_seconds"]) / positive(row["seed_push_to_emit_p99_seconds"]) for row in rows]
    scores = [f * r ** -0.25 for f, r in zip(fps, ratios)]
    if len({row["id"] for row in rows}) != len(rows): raise ValueError("duplicate case")
    if not all(math.isclose(value, positive(row["case_score"]), rel_tol=1e-9, abs_tol=0) for value, row in zip(scores, rows)):
        raise ValueError("case score mismatch")
    return {"score": gm(scores), "fps_geomean": gm(fps), "p99_geomean_ratio": gm(ratios)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", default="/app")
    parser.add_argument("--log-dir", default="/logs/verifier")
    args = parser.parse_args()
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); code = _run(args.app_dir, args.log_dir)
    detail = _json(log_dir / "sdv2_pipeline_v2_results.json", 32 * 1024 * 1024)
    reward = _json(log_dir / "reward.json", 64 * 1024)
    output = {
        "schema_version": 1, "task": "streamdiffusionv2_dual_h100_pipeline_runtime_seed_v2",
        "status": "ERROR", "valid": False, "score": 0.0, "reward": 0.0, "pass_rate": 0.0,
        "summary": "SDV2 pipeline verifier did not produce a usable result",
        "gates": {name: False for name in GATES}, "wall_seconds": round(time.monotonic() - started, 3),
        "metrics": {"source": "sdv2_pipeline_v2_verifier_adapter", "candidate_output_suppressed": True,
                    "result_protocol_records": 1, "verifier_returncode": code,
                    "primary_score_name": "latency_weighted_fps", "primary_score_unit": "fps",
                    "primary_score_direction": "maximize"},
    }
    usable = code == 0 and detail is not None and reward is not None and isinstance(reward.get("gates"), dict) and set(reward["gates"]) == set(GATES)
    if usable:
        score = reward.get("reward")
        valid = reward.get("valid") is True
        usable = isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(float(score)) and float(score) >= 0 and detail.get("valid") is valid
    if usable and valid:
        rows = detail.get("rows")
        try:
            derived = _derive_metrics(rows)
            usable = (
                all(detail.get("gates", {}).get(name) is True and reward["gates"][name] is True for name in GATES)
                # `calibrated` is set by the verifier when this run measured
                # its own seed reference on its own devices.
                and detail.get("calibrated") is True
                and all(math.isclose(value, float(detail[key]), rel_tol=1e-9, abs_tol=0) for key, value in derived.items())
                and math.isclose(derived["score"], float(score), rel_tol=1e-9, abs_tol=0)
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            usable = False
    if usable:
        output["gates"] = {name: bool(reward["gates"][name]) for name in GATES}
        output["valid"] = valid
        output["status"] = "OK" if valid else "REJECTED"
        output["score"] = output["reward"] = float(score) if valid else 0.0
        output["pass_rate"] = 1.0 if valid else 0.0
        failed = [name for name, passed in output["gates"].items() if not passed]
        output["summary"] = (f"valid: {output['score']:.6f} latency-weighted FPS" if valid else "failed gates: " + ", ".join(failed))
        if valid:
            output["metrics"].update(derived)
            output["metrics"]["rows"] = detail["rows"]
            output["metrics"]["suite"] = detail.get("suite")
            for key in ("evaluation_protocol_id", "suite_id", "suite_manifest_sha256", "formal_case_count"):
                output["metrics"][key] = detail.get(key)
            output["metrics"]["verifier_wall_seconds"] = detail.get("verifier_wall_seconds")
    print(START_MARKER); print(json.dumps(output, sort_keys=True)); print(END_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
