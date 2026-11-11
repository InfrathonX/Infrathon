#!/usr/bin/env python3
"""Collect fixed-time Flash-KMeans evaluation snapshots.

By default, the first three points are SForge's ``auto-1`` through ``auto-3``
reports (30/60/90 minutes), followed by an independent two-hour judge
evaluation. Longer fixed-budget runs can supply a different interval and final
minute. SForge's ``final_result.json`` is a best-so-far run summary and is
deliberately not accepted as the final point.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUTO_SNAPSHOTS = ((1, 30), (2, 60), (3, 90))
FINAL_MINUTES = 120
DEFAULT_FINAL_FILES = ("final-120m-report.json", "final-120m-eval.log")
_AUTO_DIR_RE = re.compile(r"auto-\d+$")
SCHEDULE_TOLERANCE_SECONDS = {
    "auto_snapshot": 120.0,
    "independent_final_eval": 60.0,
}
# Matches validate_formal_artifacts.SINGLE_GPU_MAX_JUDGE_DRAIN_SECONDS.
SINGLE_GPU_MAX_JUDGE_DRAIN_SECONDS = 7200.0


class CurveError(ValueError):
    """Raised when run artifacts cannot represent the requested curve."""


def _load_jsonish(path: Path) -> dict[str, Any]:
    """Load a JSON object, or the last report-like JSON object from a log."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CurveError(f"cannot read {path}: {exc}") from exc

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = _last_report_object(text)

    if not isinstance(value, dict):
        raise CurveError(f"{path} does not contain a JSON object")
    return value


def _looks_report_like(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if {"score", "valid", "pass_rate"}.issubset(value):
        return True
    return any(
        isinstance(value.get(key), dict)
        and {"score", "valid", "pass_rate"}.issubset(value[key])
        for key in ("report", "eval_report", "result")
    )


def _last_report_object(text: str) -> dict[str, Any]:
    """Extract the last complete report object from mixed SForge stdout."""
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for match in re.finditer(r"\{", text):
        start = match.start()
        try:
            value, relative_end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and _looks_report_like(value):
            candidates.append((start + relative_end, start, value))
    if not candidates:
        raise CurveError("no complete evaluation report JSON object found")

    # A wrapper and its nested report can both decode.  Prefer the object that
    # reaches furthest into the log, then the outer (earlier) object.
    _, _, value = max(candidates, key=lambda item: (item[0], -item[1]))
    return value


def _unwrap_report(payload: dict[str, Any], path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if {"score", "valid", "pass_rate"}.issubset(payload):
        return payload, payload
    for key in ("report", "eval_report", "result"):
        nested = payload.get(key)
        if isinstance(nested, dict) and {"score", "valid", "pass_rate"}.issubset(nested):
            return nested, payload
    raise CurveError(
        f"{path} is not an evaluation report: expected score, valid, and pass_rate"
    )


def _finite_number(value: object, *, field: str, path: Path, nullable: bool = False) -> float | int | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CurveError(f"{path}: {field} must be a finite number")
    if not math.isfinite(float(value)):
        raise CurveError(f"{path}: {field} must be a finite number")
    return value


def _validated_measurements(report: dict[str, Any], path: Path) -> dict[str, Any]:
    valid = report.get("valid")
    if not isinstance(valid, bool):
        raise CurveError(f"{path}: valid must be a boolean")

    pass_rate = _finite_number(report.get("pass_rate"), field="pass_rate", path=path)
    assert pass_rate is not None
    if not 0.0 <= float(pass_rate) <= 1.0:
        raise CurveError(f"{path}: pass_rate must be between 0 and 1")

    raw_score = _finite_number(
        report.get("score"), field="score", path=path, nullable=True
    )
    score_0_100 = _finite_number(
        report.get("score_0_100"),
        field="score_0_100",
        path=path,
        nullable=True,
    )
    return {
        "raw_score": raw_score,
        "valid": valid,
        "pass_rate": pass_rate,
        "score_0_100": score_0_100,
    }


def _parse_iso_timestamp(value: str, *, field: str, path: Path) -> float:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CurveError(f"{path}: invalid {field} timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        # SForge writes a naive local timestamp, but also writes the epoch on
        # the next line.  This branch mainly supports hand-made fixtures.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _timestamp_value(value: object, *, field: str, path: Path) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            numeric_text = float(value)
        except ValueError:
            pass
        else:
            if not math.isfinite(numeric_text):
                raise CurveError(f"{path}: {field} must be a finite timestamp")
            return numeric_text
        return _parse_iso_timestamp(value, field=field, path=path)
    numeric = _finite_number(value, field=field, path=path)
    assert numeric is not None
    return float(numeric)


def _read_started_at(task_run_dir: Path) -> float:
    path = task_run_dir / "started_at"
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    except OSError as exc:
        raise CurveError(f"cannot read SForge start time {path}: {exc}") from exc

    if len(lines) >= 2 and lines[1]:
        value = _timestamp_value(lines[1], field="started_at epoch", path=path)
    elif lines and lines[0]:
        value = _timestamp_value(lines[0], field="started_at", path=path)
    else:
        value = None
    if value is None or value <= 0:
        raise CurveError(f"{path} does not contain a positive start timestamp")
    return value


def _iso_utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_task_run_dir(run_root: Path) -> Path:
    run_root = run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise CurveError(f"run root is not a directory: {run_root}")
    if (run_root / "started_at").is_file() and (run_root / "submissions").is_dir():
        return run_root

    candidates = [
        child
        for child in run_root.iterdir()
        if child.is_dir()
        and (child / "started_at").is_file()
        and (child / "submissions").is_dir()
    ]
    if len(candidates) != 1:
        raise CurveError(
            f"{run_root} is not a task run directory and contains "
            f"{len(candidates)} candidate task directories (expected exactly one)"
        )
    return candidates[0].resolve()


def _tick_timestamps(task_run_dir: Path) -> dict[str, float]:
    path = task_run_dir / "auto_eval_ticks.log"
    if not path.is_file():
        return {}
    result: dict[str, float] = {}
    pattern = re.compile(r"^\[([^\]]+)\].*\bround=(auto-\d+)\s*$")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        match = pattern.match(line.strip())
        if not match:
            continue
        try:
            result[match.group(2)] = _parse_iso_timestamp(
                match.group(1), field="auto-eval tick", path=path
            )
        except CurveError:
            continue
    return result


def _report_timestamp(
    report: dict[str, Any],
    wrapper: dict[str, Any],
    path: Path,
    *,
    tick_timestamp: float | None = None,
    fallback_to_mtime: bool = True,
) -> tuple[float, str] | None:
    # The auto tick is the snapshot/extraction boundary. A judge may start
    # later after waiting for the single-GPU queue, so report.submitted_at is
    # not the right time coordinate for a periodic snapshot.
    if tick_timestamp is not None:
        return tick_timestamp, "auto_eval_ticks.log"
    for owner, owner_name in ((report, "report"), (wrapper, "wrapper")):
        for key in ("submitted_at", "evaluated_at", "timestamp", "completed_at"):
            if key in owner and owner[key] is not None:
                value = _timestamp_value(owner[key], field=key, path=path)
                assert value is not None
                return value, f"{owner_name}.{key}"
    if not fallback_to_mtime:
        return None
    try:
        return path.stat().st_mtime, "report_file_mtime"
    except OSError as exc:
        raise CurveError(f"cannot determine timestamp for {path}: {exc}") from exc


def _explicit_elapsed(
    report: dict[str, Any], wrapper: dict[str, Any], path: Path
) -> tuple[float, str] | None:
    for owner, owner_name in ((wrapper, "wrapper"), (report, "report")):
        for key in (
            "actual_elapsed_seconds",
            "snapshot_elapsed_seconds",
            "elapsed_seconds",
        ):
            if key in owner and owner[key] is not None:
                value = _finite_number(owner[key], field=key, path=path)
                assert value is not None
                if float(value) < 0:
                    raise CurveError(f"{path}: {key} cannot be negative")
                return float(value), f"{owner_name}.{key}"
    return None


def _explicit_snapshot_timestamp(
    wrapper: dict[str, Any], path: Path
) -> tuple[float, str] | None:
    for key in ("snapshot_captured_at", "captured_at", "snapshot_at"):
        if key in wrapper and wrapper[key] is not None:
            value = _timestamp_value(wrapper[key], field=key, path=path)
            assert value is not None
            return value, f"wrapper.{key}"
    return None


def _display_source(path: Path, run_root: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), run_root.resolve())
    except (OSError, ValueError):
        return str(path)


def _make_point(
    *,
    report_path: Path,
    report: dict[str, Any],
    wrapper: dict[str, Any],
    run_root: Path,
    started_at: float,
    scheduled_minutes: int,
    kind: str,
    round_id: str,
    tick_timestamp: float | None = None,
    allow_explicit_elapsed: bool = False,
) -> dict[str, Any]:
    measurements = _validated_measurements(report, report_path)
    elapsed_info = _explicit_elapsed(report, wrapper, report_path) if allow_explicit_elapsed else None
    timestamp_info = (
        _explicit_snapshot_timestamp(wrapper, report_path)
        if elapsed_info is not None
        else _report_timestamp(
            report,
            wrapper,
            report_path,
            tick_timestamp=tick_timestamp,
            fallback_to_mtime=True,
        )
    )
    if timestamp_info is None:
        assert elapsed_info is not None
        timestamp = started_at + elapsed_info[0]
        timestamp_source = f"started_at + {elapsed_info[1]}"
    else:
        timestamp, timestamp_source = timestamp_info
    if elapsed_info is None:
        elapsed_seconds = timestamp - started_at
        elapsed_source = f"{timestamp_source} - started_at"
    else:
        elapsed_seconds, elapsed_source = elapsed_info
    if elapsed_seconds < -1:
        raise CurveError(
            f"{report_path}: evaluation timestamp predates run start by "
            f"{-elapsed_seconds:.1f}s"
        )
    elapsed_seconds = max(0.0, elapsed_seconds)
    scheduled_seconds = scheduled_minutes * 60
    # Single-GPU lease runs seal admission at the deadline (Work paused,
    # never-resume) and may drain a Judge that was already queued before
    # freezing the container; the seal is the agent's work boundary, so the
    # schedule check anchors on it and the later freeze is only bounded.
    seal_elapsed = None
    if allow_explicit_elapsed and wrapper.get("seal_requested_elapsed_seconds") is not None:
        seal_elapsed = _finite_number(
            wrapper["seal_requested_elapsed_seconds"],
            field="seal_requested_elapsed_seconds",
            path=report_path,
        )
        assert seal_elapsed is not None
        seal_elapsed = float(seal_elapsed)
        if seal_elapsed < 0:
            raise CurveError(f"{report_path}: seal_requested_elapsed_seconds cannot be negative")
        if elapsed_seconds < seal_elapsed - 1.0:
            raise CurveError(
                f"{report_path}: {round_id} snapshot ({elapsed_seconds:.1f}s) "
                f"precedes its seal ({seal_elapsed:.1f}s)"
            )
        if elapsed_seconds - seal_elapsed > SINGLE_GPU_MAX_JUDGE_DRAIN_SECONDS:
            raise CurveError(
                f"{report_path}: {round_id} snapshot is {elapsed_seconds - seal_elapsed:.1f}s "
                f"after its seal (allowed {SINGLE_GPU_MAX_JUDGE_DRAIN_SECONDS:.0f}s)"
            )
    schedule_anchor = seal_elapsed if seal_elapsed is not None else elapsed_seconds
    schedule_delta = schedule_anchor - scheduled_seconds
    tolerance = SCHEDULE_TOLERANCE_SECONDS[kind]
    if abs(schedule_delta) > tolerance:
        what = "seal" if seal_elapsed is not None else "snapshot"
        raise CurveError(
            f"{report_path}: {round_id} {what} is {schedule_delta:+.1f}s "
            f"from its {scheduled_minutes}m target (allowed ±{tolerance:.0f}s)"
        )

    return {
        "label": f"{scheduled_minutes}m",
        "kind": kind,
        "round_id": round_id,
        "scheduled_elapsed_seconds": scheduled_seconds,
        "actual_elapsed_seconds": elapsed_seconds,
        "seal_elapsed_seconds": seal_elapsed,
        "schedule_anchor": "seal" if seal_elapsed is not None else "snapshot",
        "schedule_delta_seconds": schedule_delta,
        "timestamp_epoch": timestamp,
        "timestamp": _iso_utc(timestamp),
        "timestamp_source": timestamp_source,
        "elapsed_source": elapsed_source,
        "source_report": _display_source(report_path, run_root),
        "submission_id": report.get("submission_id"),
        **measurements,
    }


def _resolve_final_path(
    run_root: Path, final_eval: Path | None, *, final_minutes: int
) -> Path:
    if final_eval is not None:
        path = final_eval.expanduser()
        if not path.is_absolute():
            path = run_root / path
        path = path.resolve()
        if not path.is_file():
            raise CurveError(
                f"independent {final_minutes}m evaluation file not found: {path}"
            )
        return path

    default_names = (
        DEFAULT_FINAL_FILES
        if final_minutes == FINAL_MINUTES
        else (
            f"final-{final_minutes}m-report.json",
            f"final-{final_minutes}m-eval.log",
        )
    )
    for name in default_names:
        path = run_root / name
        if path.is_file():
            return path.resolve()
    names = ", ".join(default_names)
    raise CurveError(
        f"independent {final_minutes}m evaluation is required; "
        "pass --final-eval or create "
        f"one of: {names}"
    )


def _assert_independent_final(
    path: Path, payload: dict[str, Any], *, final_minutes: int
) -> None:
    if path.name == "final_result.json":
        raise CurveError(
            "final_result.json is a best-so-far run summary, not an independent "
            f"{final_minutes}m judge snapshot"
        )
    if any(_AUTO_DIR_RE.fullmatch(parent.name) for parent in path.parents):
        raise CurveError(
            f"{path} is an auto-eval report; provide the separate "
            f"{final_minutes}m final evaluation"
        )
    if not _looks_report_like(payload):
        best_keys = {"best_score", "best_round", "best_pass_rate"}
        if best_keys.intersection(payload):
            raise CurveError(
                f"{path} contains best-so-far fields rather than an evaluation report"
            )


def collect_curve(
    run_root: Path,
    final_eval: Path | None = None,
    *,
    auto_snapshots: tuple[tuple[int, int], ...] = AUTO_SNAPSHOTS,
    final_minutes: int = FINAL_MINUTES,
) -> dict[str, Any]:
    """Return curve data without writing files."""
    if final_minutes <= 0:
        raise CurveError("final_minutes must be positive")
    original_root = run_root.expanduser().resolve()
    task_run_dir = _resolve_task_run_dir(original_root)
    started_at = _read_started_at(task_run_dir)
    tick_times = _tick_timestamps(task_run_dir)

    points: list[dict[str, Any]] = []
    task_id: str | None = None
    for sequence, minutes in auto_snapshots:
        round_id = f"auto-{sequence}"
        report_path = task_run_dir / "submissions" / round_id / "report.json"
        if not report_path.is_file():
            raise CurveError(
                f"missing required {minutes}m snapshot: {report_path}"
            )
        payload = _load_jsonish(report_path)
        report, wrapper = _unwrap_report(payload, report_path)
        report_task = report.get("task_id")
        if isinstance(report_task, str):
            if task_id is not None and report_task != task_id:
                raise CurveError(
                    f"{report_path}: task_id {report_task!r} does not match {task_id!r}"
                )
            task_id = report_task
        points.append(
            _make_point(
                report_path=report_path,
                report=report,
                wrapper=wrapper,
                run_root=original_root,
                started_at=started_at,
                scheduled_minutes=minutes,
                kind="auto_snapshot",
                round_id=round_id,
                tick_timestamp=tick_times.get(round_id),
            )
        )

    final_path = _resolve_final_path(
        task_run_dir, final_eval, final_minutes=final_minutes
    )
    final_payload = _load_jsonish(final_path)
    _assert_independent_final(
        final_path, final_payload, final_minutes=final_minutes
    )
    final_report, final_wrapper = _unwrap_report(final_payload, final_path)
    final_task = final_report.get("task_id")
    if isinstance(final_task, str):
        if task_id is not None and final_task != task_id:
            raise CurveError(
                f"{final_path}: task_id {final_task!r} does not match {task_id!r}"
            )
        task_id = final_task
    points.append(
        _make_point(
            report_path=final_path,
            report=final_report,
            wrapper=final_wrapper,
            run_root=original_root,
            started_at=started_at,
            scheduled_minutes=final_minutes,
            kind="independent_final_eval",
            round_id=f"final-{final_minutes}m",
            allow_explicit_elapsed=True,
        )
    )

    return {
        "schema_version": 1,
        "task_id": task_id or task_run_dir.name,
        "run_root": str(original_root),
        "task_run_dir": str(task_run_dir),
        "run_started_at_epoch": started_at,
        "run_started_at": _iso_utc(started_at),
        "point_semantics": (
            "raw fixed-time snapshots; values are not cumulative best-so-far"
        ),
        "final_point_policy": (
            "independent judge evaluation; final_result.json is never used"
        ),
        "points": points,
    }


def _format_number(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(curve: dict[str, Any]) -> str:
    lines = [
        f"# Evaluation curve: {curve['task_id']}",
        "",
        "These are raw fixed-time snapshots, not cumulative best scores. "
        f"The {curve['points'][-1]['label']} row is a separate judge evaluation.",
        "",
        "| target | actual | timestamp (UTC) | source | valid | pass rate | raw score | score 0–100 |",
        "| ---: | ---: | --- | --- | :---: | ---: | ---: | ---: |",
    ]
    for point in curve["points"]:
        source = str(point["source_report"]).replace("|", "\\|")
        elapsed = float(point["actual_elapsed_seconds"]) / 60.0
        lines.append(
            "| {label} | {elapsed:.2f}m | {timestamp} | `{source}` | "
            "{valid} | {pass_rate} | {raw_score} | {score_0_100} |".format(
                label=point["label"],
                elapsed=elapsed,
                timestamp=point["timestamp"],
                source=source,
                valid=_format_number(point["valid"]),
                pass_rate=_format_number(point["pass_rate"]),
                raw_score=_format_number(point["raw_score"]),
                score_0_100=_format_number(point["score_0_100"]),
            )
        )
    return "\n".join(lines) + "\n"


def write_curve(
    curve: dict[str, Any],
    output_dir: Path,
    *,
    json_name: str = "curve.json",
    markdown_name: str = "curve.md",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / json_name
    markdown_path = output_dir / markdown_name
    json_path.write_text(
        json.dumps(curve, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(curve), encoding="utf-8")
    return json_path, markdown_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_root",
        type=Path,
        help="SForge task run directory (or a run-id directory containing one task)",
    )
    parser.add_argument(
        "final_eval_positional",
        nargs="?",
        type=Path,
        help="independent 120m report/log (legacy positional form)",
    )
    parser.add_argument(
        "--final-eval",
        type=Path,
        help=(
            "independent final report JSON or sforge eval --json log; defaults "
            "to the final-<minutes>m report/log in the task run"
        ),
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=30,
        help="minutes between auto snapshots (default: 30)",
    )
    parser.add_argument(
        "--no-auto-snapshots",
        action="store_true",
        help=(
            "build the curve from the independent final evaluation only; "
            "required form for runs launched with --disable-auto-eval, whose "
            "periodic states are saved to the version store for later replay "
            "instead of being judged as auto-N during the run"
        ),
    )
    parser.add_argument(
        "--final-minutes",
        type=int,
        default=FINAL_MINUTES,
        help="fixed-budget final snapshot minute (default: 120)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: the run_root argument)",
    )
    parser.add_argument("--json-name", default="curve.json")
    parser.add_argument("--markdown-name", default="curve.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.final_eval is not None and args.final_eval_positional is not None:
        print(
            "error: give the final evaluation either positionally or with "
            "--final-eval, not both",
            file=sys.stderr,
        )
        return 2
    final_eval = args.final_eval or args.final_eval_positional
    if not args.no_auto_snapshots and (
        args.interval_minutes <= 0
        or args.final_minutes <= args.interval_minutes
        or args.final_minutes % args.interval_minutes != 0
    ):
        print(
            "error: final minutes must be an exact multiple greater than "
            "the positive interval minutes",
            file=sys.stderr,
        )
        return 2
    auto_snapshots = (
        ()
        if args.no_auto_snapshots
        else tuple(
            (sequence, minutes)
            for sequence, minutes in enumerate(
                range(
                    args.interval_minutes,
                    args.final_minutes,
                    args.interval_minutes,
                ),
                start=1,
            )
        )
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else args.run_root.expanduser().resolve()
    )
    try:
        curve = collect_curve(
            args.run_root,
            final_eval,
            auto_snapshots=auto_snapshots,
            final_minutes=args.final_minutes,
        )
        json_path, markdown_path = write_curve(
            curve,
            output_dir,
            json_name=args.json_name,
            markdown_name=args.markdown_name,
        )
    except (CurveError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
