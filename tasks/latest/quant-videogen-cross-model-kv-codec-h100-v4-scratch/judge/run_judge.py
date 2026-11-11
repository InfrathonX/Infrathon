#!/usr/bin/env python3
"""Fail-closed public adapter for the root-only protected evaluator.

Only aggregate validity/score and a generic status cross the Judge-to-Work
boundary.  Case ids, model names, gates, thresholds, metrics, raw logs,
release metadata, and controller errors remain in the root-private artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


START = ">>>>> Start Structured Result"
END = ">>>>> End Structured Result"
TASK_ID = "quant_videogen_cross_model_kv_codec_h100_scratch_v4"
RESULT_SCHEMA_VERSION = 5
RELEASE_PATH = Path("/opt/mlsysforge/configs/h100_scratch_v4_release.json")
# This device class's quality reference (the zero point of q), staged beside
# the release from hw/<type>/judge/ by scripts/build_images.sh.
QUALITY_PATH = Path("/opt/mlsysforge/configs/h100_scratch_v4_quality.json")
CONTROLLER = Path("/opt/qvg-judge/evaluate_hidden.py")
# Used only when the release itself is unreadable; the image creates exactly
# these slot accounts (qvg-candidate, qvg-candidate-2..8; UID == GID).
FALLBACK_SLOT_UIDS = tuple(range(23001, 23009))
EXIT_CODES = {"evaluated": 0, "candidate_failure": 2, "infrastructure_failure": 3}


def emit(payload: dict[str, Any]) -> None:
    print(START)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    print(END)


def _strict_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("private result is missing, unsafe, or oversized")
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON number: {token}")
        ),
    )


def _controller_module() -> Any:
    """The controller module itself, so both sides derive one timeout."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import evaluate_hidden

    return evaluate_hidden


def controller_timeout_seconds(release_path: Path, gpu_count: int) -> int:
    """Derive the controller bound from the budgets the controller enforces.

    A constant smaller than the stage and metric budgets below it turns a slow
    candidate - or a slower device - into an infrastructure failure instead of
    the verdict the controller was about to produce, so the adapter calls the
    very function ``evaluate_hidden`` records: the list-scheduling bound of
    the static plan on this Judge's GPU count.
    """
    controller = _controller_module()
    return controller.controller_budget_seconds(controller.load_release(release_path), gpu_count)


def validate_private_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("private result must be an object")
    if value.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("private result schema mismatch")
    if value.get("task_id") != TASK_ID:
        raise ValueError("private result task mismatch")
    if not isinstance(value.get("valid"), bool):
        raise ValueError("private validity must be boolean")
    score = value.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("private score must be numeric")
    score = float(score)
    if not math.isfinite(score):
        raise ValueError("private score must be finite")
    outcome = value.get("outcome")
    if outcome not in EXIT_CODES:
        raise ValueError("private outcome is invalid")
    candidate_invalid = value.get("candidate_invalid")
    infrastructure_invalid = value.get("infrastructure_invalid")
    if not isinstance(candidate_invalid, bool) or not isinstance(infrastructure_invalid, bool):
        raise ValueError("private classifications must be boolean")
    if candidate_invalid and infrastructure_invalid:
        raise ValueError("private classifications conflict")
    if outcome == "candidate_failure" and not candidate_invalid:
        raise ValueError("candidate failure is misclassified")
    if outcome == "infrastructure_failure" and not infrastructure_invalid:
        raise ValueError("infrastructure failure is misclassified")
    uncalibrated = value.get("reason") == "uncalibrated_release"
    if outcome == "evaluated":
        if uncalibrated:
            # A calibration run: every case ran and was recorded, but the
            # release has no frozen gates, so no verdict may leave the Judge.
            if value["valid"] or candidate_invalid or not infrastructure_invalid:
                raise ValueError("uncalibrated result is misclassified")
        elif infrastructure_invalid or candidate_invalid != (not value["valid"]):
            raise ValueError("evaluated result is misclassified")
    if value["valid"] and (candidate_invalid or infrastructure_invalid or outcome != "evaluated"):
        raise ValueError("valid result carries a failure classification")
    if not value["valid"] and score != 0.0:
        raise ValueError("invalid result has a non-zero score")
    if value["valid"] and score <= 0.0:
        raise ValueError("valid score is not positive")
    value["score"] = score
    return value


def public_payload(detail: dict[str, Any]) -> dict[str, Any]:
    score = float(detail["score"])
    uncalibrated = detail["outcome"] == "evaluated" and detail.get("reason") == "uncalibrated_release"
    status = "uncalibrated" if uncalibrated else {
        "evaluated": "completed",
        "candidate_failure": "candidate_error",
        "infrastructure_failure": "infrastructure_error",
    }[detail["outcome"]]
    payload: dict[str, Any] = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "valid": bool(detail["valid"]),
        "score": score,
        "primary_score_name": "quality_weighted_persistent_kv_compression_score",
        "primary_score_value": score,
        "candidate_invalid": bool(detail["candidate_invalid"]),
        "infrastructure_invalid": bool(detail["infrastructure_invalid"]),
        "metrics": {
            "candidate_invalid": bool(detail["candidate_invalid"]),
            "infrastructure_invalid": bool(detail["infrastructure_invalid"]),
            "evaluation_status": status,
        },
    }
    if detail["outcome"] == "candidate_failure":
        payload["error"] = "the protected evaluation did not accept this submission"
    elif detail["outcome"] == "infrastructure_failure":
        payload["error"] = "the protected evaluation was unavailable"
    elif uncalibrated:
        payload["error"] = "the protected evaluation is not calibrated yet"
    return payload


def _fallback_private(path: Path, reason: str, **extra: Any) -> dict[str, Any]:
    detail = {
        **extra,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "outcome": "infrastructure_failure",
        "valid": False,
        "candidate_invalid": False,
        "infrastructure_invalid": True,
        "score": 0.0,
        "error": reason,
    }
    path.write_text(json.dumps(detail, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    return detail


def _progress(log_dir: Path) -> Any:
    """The controller's last per-case snapshot, for a controller that was killed."""
    try:
        return _strict_json(log_dir / "qvg-progress.json")
    except Exception:
        return None


def _pkill_uids(uids: Sequence[int]) -> None:
    try:
        subprocess.run(
            ["pkill", "-KILL", "-u", ",".join(str(uid) for uid in uids)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def kill_controller_tree(session_id: int, slot_uids: Sequence[int]) -> None:
    """Kill the controller, every trusted child in its session, and every
    candidate process of every slot account (candidates lead their own
    sessions, so only their UIDs reach them)."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.getsid(int(entry.name)) == session_id:
                os.kill(int(entry.name), signal.SIGKILL)
        except OSError:
            pass
    _pkill_uids(slot_uids)


def run_controller(
    command: list[str], *, log: Path, timeout: int, slot_uids: Sequence[int]
) -> int:
    """Run the controller in its own session; on its deadline (or any adapter
    error) kill the whole tree and re-raise."""
    with log.open("w", encoding="utf-8") as handle:
        os.chmod(log, 0o600)
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            kill_controller_tree(process.pid, slot_uids)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.submission != Path("/workspace/submission") or args.log_dir != Path("/logs/verifier"):
        emit(public_payload(_fallback_private(Path("/tmp/qvg-invalid-invocation.json"), "invalid invocation")))
        return

    args.log_dir.mkdir(parents=True, exist_ok=True)
    os.chown(args.log_dir, 0, 0)
    os.chmod(args.log_dir, 0o700)
    private_path = args.log_dir / "qvg-private.json"
    controller_log = args.log_dir / "controller.log"
    # Never read back a previous run's verdict or snapshot as this run's.
    private_path.unlink(missing_ok=True)
    (args.log_dir / "qvg-progress.json").unlink(missing_ok=True)
    slot_uids: Sequence[int] = FALLBACK_SLOT_UIDS
    try:
        controller = _controller_module()
        quality = QUALITY_PATH if QUALITY_PATH.exists() else None
        release = controller.load_release(RELEASE_PATH, quality_path=quality)
        slot_uids = release["judge"]["slot_uids"]
        gpus, _fleet = controller.evaluator_hardware(release)
        gpu_count = len(gpus)
        controller_timeout = controller.controller_budget_seconds(release, gpu_count)
    except Exception as exc:
        emit(public_payload(_fallback_private(
            private_path,
            f"the frozen release or the Judge GPUs are unusable: {type(exc).__name__}: {exc}",
        )))
        return
    try:
        returncode = run_controller(
            [
                sys.executable,
                str(CONTROLLER),
                "--submission",
                str(args.submission),
                "--release",
                str(RELEASE_PATH),
                *(["--quality", str(quality)] if quality is not None else []),
                "--output",
                str(private_path),
                "--gpu-count",
                str(gpu_count),
            ],
            log=controller_log,
            timeout=controller_timeout,
            slot_uids=slot_uids,
        )
    except Exception as exc:
        reason = (
            f"protected controller exceeded its {controller_timeout} s budget"
            if isinstance(exc, subprocess.TimeoutExpired)
            else f"protected controller could not complete: {type(exc).__name__}"
        )
        detail = _fallback_private(
            private_path,
            reason,
            controller_budget_seconds=controller_timeout,
            gpu_count=gpu_count,
            progress=_progress(args.log_dir),
        )
        emit(public_payload(detail))
        return

    try:
        detail = validate_private_result(_strict_json(private_path))
        if returncode != EXIT_CODES[detail["outcome"]]:
            raise ValueError("controller return code disagrees with its private result")
    except Exception as exc:
        detail = _fallback_private(
            private_path,
            f"protected controller returned an invalid result: {type(exc).__name__}",
        )
    emit(public_payload(detail))


if __name__ == "__main__":
    main()
