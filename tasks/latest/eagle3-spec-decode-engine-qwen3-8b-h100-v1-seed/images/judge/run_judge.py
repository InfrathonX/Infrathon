#!/usr/bin/env python3
"""Run the private EAGLE-3 verifier and emit one redacted result record."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
from typing import Any, Mapping

MARKER_START = ">>>>> Start Structured Result"
MARKER_END = ">>>>> End Structured Result"
DETAIL_NAME = "eagle3_spec_decode_private.json"
RESULT_FILES = ("reward.json", "reward.txt", DETAIL_NAME, "verifier.log")
MAX_RESULT_BYTES = 16 * 1024 * 1024
GATES = (
    "protected_ok",
    "source_ok",
    "worker_ok",
    "o1_ok",
    "o2_ok",
    "tiers_ok",
    "o3_o4_ok",
    "o5_ok",
    "wall_ok",
    "memory_ok",
    "idle_ok",
)
GATE_LABELS = {
    "protected_ok": "protected Judge/runtime identities",
    "source_ok": "submission source admission",
    "worker_ok": "isolated worker protocol",
    "o1_ok": "teacher-forced hard-margin oracle",
    "o2_ok": "global argmax-disagreement rate",
    "tiers_ok": "tiered margin-CDF oracle",
    "o3_o4_ok": "termination, shape, range, and prefix contract",
    "o5_ok": "determinism and request isolation",
    "wall_ok": "per-cell wall caps",
    "memory_ok": "per-cell peak-memory caps",
    "idle_ok": "post-call CUDA idleness",
}


def reject_json_constant(value: str) -> None:
    """Reject NaN and infinity in private records."""
    raise ValueError(f"non-finite JSON constant: {value}")


def finite_tree(value: Any) -> bool:
    """Return whether every numeric leaf can be represented finitely."""
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        try:
            return math.isfinite(float(value))
        except (OverflowError, ValueError):
            return False
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return False


def finite_number(value: Any) -> float | None:
    """Normalize a non-boolean finite JSON number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one bounded regular result, refusing links and malformed JSON."""
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink != 1
            or not 0 < metadata.st_size <= MAX_RESULT_BYTES
        ):
            return {}
        parsed = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) and finite_tree(parsed) else {}


def prepare_log_dir(log_dir: Path) -> None:
    """Remove stale result records before launching the verifier."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dir.chmod(0o700)
    for name in RESULT_FILES:
        path = log_dir / name
        try:
            if path.exists() or path.is_symlink():
                path.unlink()
        except OSError:
            pass


def process_table() -> dict[int, tuple[int, int]]:
    """Snapshot Linux PID to parent-PID/process-group relationships."""
    table: dict[int, tuple[int, int]] = {}
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError:
        return table
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            fields = raw.rsplit(")", 1)[1].split()
            process_id = int(entry.name)
            table[process_id] = (int(fields[1]), int(fields[2]))
        except (IndexError, OSError, ProcessLookupError, ValueError):
            continue
    return table


def process_tree_groups(root_pid: int) -> set[int]:
    """Return all groups below root, including separately created sessions."""
    table = process_table()
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for process_id, (parent_id, _group_id) in table.items():
            if parent_id in descendants and process_id not in descendants:
                descendants.add(process_id)
                changed = True
    groups = {
        table[process_id][1]
        for process_id in descendants
        if process_id in table
    }
    try:
        groups.add(os.getpgid(root_pid))
    except (OSError, ProcessLookupError):
        pass
    return groups


def signal_groups(groups: set[int], requested: signal.Signals) -> None:
    """Signal a stable snapshot of process groups, newest leaders first."""
    for group in sorted(groups, reverse=True):
        try:
            os.killpg(group, requested)
        except (OSError, ProcessLookupError):
            pass


def terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate and then kill descendants even if they created sessions."""
    groups = process_tree_groups(process.pid)
    signal_groups(groups, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    groups.update(process_tree_groups(process.pid))
    signal_groups(groups, signal.SIGKILL)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def matching(
    reward: Mapping[str, Any],
    detail: Mapping[str, Any],
    key: str,
) -> float | None:
    """Accept a metric only when independent records agree exactly."""
    left = finite_number(reward.get(key))
    right = finite_number(detail.get(key))
    if left is None or right is None:
        return None
    if not math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-12):
        return None
    return left


def gate_detail(name: str, value: float | None) -> dict[str, Any]:
    """Build one redacted, user-visible gate status."""
    if value == 1.0:
        status = "PASSED"
    elif value == 0.0:
        status = "FAILED"
    else:
        status = "ERROR"
    return {
        "name": name,
        "status": status,
        "score": 1.0 if status == "PASSED" else 0.0,
        "message": f"{GATE_LABELS[name]} {status.lower()}.",
    }


EXPECTED_CELLS = {"chat_bs1", "math_bs1", "code_bs1", "longctx_bs1", "mixed_bs8", "think_bs1"}


def valid_measurements(detail):
    """Reconcile the exact trusted profile geometry and per-prompt token counts."""
    profile = detail.get("profile")
    all_anchor = detail.get("all_anchor")
    if profile not in {"anchor", "final"} or not isinstance(all_anchor, bool):
        return False
    cells, rates = detail.get("cells"), detail.get("cell_tokens_per_s")
    if not isinstance(cells, Mapping) or not isinstance(rates, Mapping):
        return False
    if set(cells) != EXPECTED_CELLS or set(rates) != EXPECTED_CELLS:
        return False
    for name in EXPECTED_CELLS:
        cell = cells[name]
        if not isinstance(cell, Mapping):
            return False
        batch = 8 if name == "mixed_bs8" else 1
        count = (5 if batch == 8 else 20) if profile == "final" else ((2 if all_anchor else 1) if batch == 8 else (8 if all_anchor else 6))
        measurements = cell.get("measurements")
        if not isinstance(measurements, list) or len(measurements) != count:
            return False
        lengths, walls = [], []
        budget = 1024 if name == "think_bs1" else 256 if name == "longctx_bs1" else 512
        for row in measurements:
            if not isinstance(row, Mapping):
                return False
            wall = finite_number(row.get("wall_s"))
            generated = row.get("generated_lengths")
            if wall is None or wall <= 0 or not isinstance(generated, list) or len(generated) != batch:
                return False
            if any(type(length) is not int or not 1 <= length <= budget for length in generated):
                return False
            lengths.extend(generated)
            walls.append(wall)
        if cell.get("generated_lengths") != lengths or type(cell.get("generated_tokens")) is not int or cell["generated_tokens"] != sum(lengths):
            return False
        rate = sum(lengths) / math.fsum(walls)
        for recorded in (cell.get("tokens_per_s"), rates[name]):
            value = finite_number(recorded)
            if value is None or value <= 0 or not math.isclose(rate, value, rel_tol=1e-9, abs_tol=1e-12):
                return False
    return True


def build_structured_result(
    reward_data: Mapping[str, Any],
    detail_data: Mapping[str, Any],
    *,
    verifier_completed: bool,
    verifier_returncode: int | None,
) -> dict[str, Any]:
    """Validate private records and retain only safe aggregate fields."""
    status = detail_data.get("status")
    pending = status == "PENDING_QUALIFICATION"
    reward = matching(reward_data, detail_data, "reward")
    reported = matching(reward_data, detail_data, "score_tokens_per_s")
    gate_values = {name: matching(reward_data, detail_data, name) for name in GATES}
    cells = detail_data.get("cell_tokens_per_s")
    cell_values = []
    if isinstance(cells, Mapping):
        for value in cells.values():
            number = finite_number(value)
            if number is None or number <= 0:
                cell_values = []
                break
            cell_values.append(number)
    recomputed = (
        math.exp(math.fsum(math.log(value) for value in cell_values) / len(cell_values))
        if cell_values
        else None
    )
    aggregate_ok = (
        reported is not None
        and recomputed is not None
        and math.isclose(reported, recomputed, rel_tol=1.0e-9, abs_tol=1.0e-12)
    )
    gates_formed = all(value in (0.0, 1.0) for value in gate_values.values())
    schema_ok = matching(reward_data, detail_data, "schema_version") == 1.0
    valid = bool(
        verifier_completed
        and verifier_returncode == 0
        and status == "PASS"
        and schema_ok
        and aggregate_ok
        and valid_measurements(detail_data)
        and reward is not None
        and math.isclose(reward, reported or 0.0, rel_tol=1.0e-12, abs_tol=1.0e-12)
        and gates_formed
        and all(value == 1.0 for value in gate_values.values())
    )
    trusted_failure = bool(
        verifier_completed
        and verifier_returncode == 1
        and schema_ok
        and gates_formed
        and any(value == 0.0 for value in gate_values.values())
        and reward == 0.0
        and reported == 0.0
    )
    if pending:
        summary = "PENDING_QUALIFICATION: helper D must freeze image identities and caps."
        valid = False
    elif valid:
        summary = f"EAGLE-3 decode verification passed; score {recomputed:.3f} tokens/s."
    elif trusted_failure:
        summary = "EAGLE-3 decode submission failed one or more required gates."
    else:
        summary = "EAGLE-3 verifier did not produce a trusted result."
    visible = gate_values if valid or trusted_failure else {name: None for name in GATES}
    accepted = float(recomputed) if valid and recomputed is not None else 0.0
    details = [gate_detail(name, visible[name]) for name in GATES]
    details.append(
        {
            "name": "performance",
            "status": "PASSED" if valid else "FAILED" if trusted_failure else "ERROR",
            "score": accepted,
            "message": "Throughput is accepted only after every required gate passes.",
        }
    )
    metrics: dict[str, Any] = {
        "source": "eagle3_spec_decode_v1_verifier_adapter",
        "adapter_schema_version": 1,
        "candidate_output_suppressed": True,
        "result_protocol_records": 1,
        "verifier_completed": verifier_completed,
        "verifier_returncode": verifier_returncode,
        "primary_score_name": "score_tokens_per_s",
        "primary_score_unit": "tokens/s",
        "primary_score_direction": "maximize",
        "score_tokens_per_s": accepted,
    }
    if valid or trusted_failure:
        metrics.update({name: gate_values[name] for name in GATES})
        for key in ("evaluation_profile", "round_id"):
            if isinstance(detail_data.get(key), str):
                metrics[key] = detail_data[key]
    if valid:
        pass_rate = 1.0
    elif trusted_failure:
        # SForge counts the performance row as a failed test whenever any
        # required gate fails. Keep the summary rate consistent with the
        # emitted 11 gate rows plus that performance row.
        pass_rate = sum(value == 1.0 for value in gate_values.values()) / (
            len(GATES) + 1
        )
    else:
        pass_rate = 0.0
    return {
        "valid": valid,
        "score": accepted,
        "pass_rate": pass_rate,
        "summary": summary,
        "details": details,
        "metrics": metrics,
    }


def evaluate(
    verifier: Path,
    app_dir: Path,
    log_dir: Path,
    timeout: float,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run the verifier in a killable session and suppress its output."""
    prepare_log_dir(log_dir)
    environment = dict(os.environ if environ is None else environ)
    completed = False
    returncode: int | None = None
    log_path = log_dir / "verifier.log"
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(verifier),
                    "--app-dir",
                    str(app_dir),
                    "--log-dir",
                    str(log_dir),
                ],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=timeout)
                completed = True
                signal_groups(process_tree_groups(process.pid), signal.SIGKILL)
            except subprocess.TimeoutExpired:
                terminate_process_tree(process)
    except (OSError, subprocess.SubprocessError):
        if "process" in locals():
            terminate_process_tree(process)
    return build_structured_result(
        read_json_object(log_dir / "reward.json"),
        read_json_object(log_dir / DETAIL_NAME),
        verifier_completed=completed,
        verifier_returncode=returncode,
    )


def emit_structured(result: Mapping[str, Any]) -> None:
    """Emit exactly one structured marker pair."""
    print(MARKER_START)
    print(json.dumps(dict(result), sort_keys=True, allow_nan=False))
    print(MARKER_END)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse stable Judge-adapter arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verifier",
        type=Path,
        default=Path("/opt/eagle3_judge/verify_eagle3.py"),
    )
    parser.add_argument("--app-dir", type=Path, default=Path(os.environ.get("APP_DIR", "/app")))
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(os.environ.get("LOG_DIR", "/logs/verifier")),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("EAGLE3_FINAL_JUDGE_TIMEOUT", "5400")),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the private verifier and always leave protocol parsing to SForge."""
    args = parse_args(argv)
    result = evaluate(
        args.verifier.resolve(),
        args.app_dir.resolve(),
        args.log_dir.resolve(),
        args.timeout,
    )
    emit_structured(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
