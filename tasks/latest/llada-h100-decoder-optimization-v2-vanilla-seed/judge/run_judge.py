#!/usr/bin/env python3
"""Run the private verifier and emit one redacted structured result."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


MARKER_START = ">>>>> Start Structured Result"
MARKER_END = ">>>>> End Structured Result"
RESULT_FILES = ("reward.json", "decoder_private.json")
EVALUATION_CONTEXT_PATH = Path("/tmp/.sforge-evaluation-context.json")


def context_timeout(round_id: str | None = None) -> float:
    if round_id is None:
        context = read_object(EVALUATION_CONTEXT_PATH)
        value = context.get("round_id")
        round_id = value.strip() if isinstance(value, str) else ""
    variable = "LLADA_FINAL_JUDGE_TIMEOUT"
    raw = os.environ.get(variable, "").strip()
    if not raw:
        raise ValueError(f"{variable} is required")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError(f"{variable} must be a positive finite number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{variable} must be a positive finite number")
    return timeout


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def prepare(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dir.chmod(0o700)
    for name in RESULT_FILES:
        path = log_dir / name
        if path.is_file() or path.is_symlink():
            path.unlink()


def gate(name: str, passed: bool, label: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASSED" if passed else "FAILED",
        "score": 1.0 if passed else 0.0,
        "message": f"{label} {'passed' if passed else 'failed'}.",
    }


def build_result(
    reward: dict[str, Any],
    detail: dict[str, Any],
    completed: bool,
    returncode: int | None,
) -> dict[str, Any]:
    statuses = {
        name: finite(reward.get(name)) == 1.0
        for name in ("source_ok", "integrity_ok", "performance_ok")
    }
    quality_ok = finite(reward.get("quality_ok")) == 1.0
    quality_enforced = detail.get("quality_hard_gate") is True
    # A host-selected quality-control run must return its ACCEPT/REJECT
    # classification even for a deliberately degraded candidate; the verifier
    # marks it non-release evidence, so its quality never gates validity here.
    quality_control = detail.get("quality_control") is True
    pilot = detail.get("pilot") is True
    score = finite(reward.get("reward"))
    valid = bool(
        completed
        and returncode == 0
        and score is not None
        and score > 0
        and all(statuses.values())
        and (quality_ok or quality_control or not quality_enforced)
    )
    accepted_score = float(score) if valid and score is not None else 0.0
    details = [
        gate("source", statuses["source_ok"], "Submission source admission"),
        gate("integrity", statuses["integrity_ok"], "Runtime integrity"),
        gate("performance", statuses["performance_ok"], "Throughput measurement"),
    ]
    # Keep the Agent-visible result deliberately small.  The root-owned
    # artifact retains detailed provenance; interactive submissions receive
    # only the accepted objective, three gates, and the quality disposition.
    metrics: dict[str, Any] = {"reward": accepted_score}
    return {
        "valid": valid,
        "score": accepted_score,
        "pass_rate": 1.0 if valid else 0.0,
        "summary": (
            f"LLaDA decoder verification passed; throughput score {accepted_score:.6f} tok/s. "
            + (
                "Quality-control classification recorded (non-release evidence)."
                if quality_control
                else
                "Semantic quality gate passed (enforced)."
                if quality_enforced
                else
                "Semantic quality diagnostic passed (non-blocking)."
                if quality_ok
                else (
                    "Semantic quality diagnostic reported issues (non-blocking)."
                    if pilot
                    else "Semantic quality diagnostic was below the reference (non-blocking)."
                )
            )
            if valid
            else "LLaDA decoder submission did not produce an accepted result."
        ),
        "details": details,
        "metrics": metrics,
    }


def evaluate(verifier: Path, app_dir: Path, log_dir: Path, timeout: float) -> dict[str, Any]:
    prepare(log_dir)
    env = dict(os.environ)
    completed = False
    returncode = None
    try:
        process = subprocess.run(
            [
                sys.executable,
                str(verifier),
                "--app-dir",
                str(app_dir),
                "--log-dir",
                str(log_dir),
            ],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        completed = True
        returncode = process.returncode
    except (OSError, subprocess.SubprocessError):
        pass
    return build_result(
        read_object(log_dir / "reward.json"),
        read_object(log_dir / "decoder_private.json"),
        completed,
        returncode,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", type=Path, default=Path(os.environ.get("APP_DIR", "/app")))
    parser.add_argument("--log-dir", type=Path, default=Path(os.environ.get("LOG_DIR", "/logs/verifier")))
    parser.add_argument(
        "--verifier",
        type=Path,
        default=Path("/opt/decoder_judge/verifier/verify_decoder.py"),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=context_timeout(),
    )
    args = parser.parse_args()
    try:
        result = evaluate(args.verifier, args.app_dir, args.log_dir, args.timeout)
    except Exception:
        result = build_result({}, {}, False, None)
    print(MARKER_START)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    print(MARKER_END)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
