#!/usr/bin/env python3
"""Flux MoE replay adapter: dual GPUs, minimal version archives, rich curves."""
from __future__ import annotations

import copy
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any, Mapping

# Importing the task adapter installs the dual-H100 Docker backend. Its managed
# Work/Judge wrapper is intentionally not installed: replay has no live Work
# container and replay_candidates.sh owns both GPU locks for the entire batch.
import sforge_dual_h100_entry  # noqa: F401
from sforge import cli
from sforge.harness import candidate_replay


METRIC_KEYS = (
    "score_tflops",
    "tp8_tflops",
    "tp4_tflops",
    "tp2_tflops",
    "ep_tflops",
    "decode_tflops",
    "sequence_tflops",
    "hidden_comm",
)
TASK_ID = "flux_moe_ag_scatter_grouped_gemm_dual_h100_seed_v1"


def _minimal_version_archive(
    archive: bytes, *, selector_file: str, version_number: int
) -> bytes:
    """Keep only the selected Flux kernel plus its two root metadata files."""

    if selector_file != "FINAL_VERSION" or not 1 <= version_number <= 999:
        raise candidate_replay.CandidateReplayError(
            "Flux replay selector must be FINAL_VERSION in the range 1..999"
        )
    candidate = f"candidates/kernel_v{version_number:03d}/submission.py"
    wanted = (candidate, "notes.md")
    selected: dict[str, tuple[tarfile.TarInfo, bytes]] = {}
    selector_info: tarfile.TarInfo | None = None
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        for member in source.getmembers():
            if not member.isfile():
                continue
            if member.name == selector_file:
                selector_info = copy.copy(member)
            if member.name not in wanted:
                continue
            stream = source.extractfile(member)
            if stream is not None:
                selected[member.name] = (copy.copy(member), stream.read())
    missing = [name for name in wanted if name not in selected]
    if missing or selector_info is None:
        raise candidate_replay.CandidateReplayError(
            "Flux per-version source archive is missing: "
            + ", ".join([*missing, *([] if selector_info else [selector_file])])
        )

    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        member, payload = selected[candidate]
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
        selector_payload = f"{version_number}\n".encode("ascii")
        selector_info.size = len(selector_payload)
        bundle.addfile(selector_info, io.BytesIO(selector_payload))
        member, payload = selected["notes.md"]
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    return output.getvalue()


def _metric_values(report: Mapping[str, Any] | None) -> dict[str, Any]:
    metrics = report.get("metrics") if isinstance(report, Mapping) else None
    if not isinstance(metrics, Mapping):
        metrics = {}
    return {key: metrics.get(key) for key in METRIC_KEYS}


def _install_rich_candidate_curve() -> None:
    original = candidate_replay.build_curve

    def build_curve(*args: Any, **kwargs: Any) -> dict[str, Any]:
        curve = original(*args, **kwargs)
        measurements = kwargs.get("measurements")
        if not isinstance(measurements, Mapping):
            return curve
        for point in curve.get("points", []):
            measurement = measurements.get(point.get("candidate_id"))
            report = (
                measurement.get("report")
                if isinstance(measurement, Mapping)
                else None
            )
            point.update(_metric_values(report))
            if point.get("score") is not None:
                point["score_tflops"] = point["score"]
        return curve

    candidate_replay.build_curve = build_curve


def _install_rich_version_curve() -> None:
    original = candidate_replay.write_measurement

    def write_measurement(path: Path, payload: Mapping[str, Any]) -> None:
        if payload.get("kind") == "sforge-version-curve":
            for point in payload.get("points", []):
                implementation_id = point.get("implementation_id")
                report = None
                if isinstance(implementation_id, str):
                    measurement_path = (
                        path.parent / implementation_id / "measurement.json"
                    )
                    try:
                        measurement = json.loads(
                            measurement_path.read_text(encoding="utf-8")
                        )
                    except (OSError, ValueError):
                        measurement = None
                    if isinstance(measurement, Mapping):
                        report = measurement.get("report")
                point.update(_metric_values(report))
                if point.get("score") is not None:
                    point["score_tflops"] = point["score"]
        original(path, payload)

    candidate_replay.write_measurement = write_measurement


def _print_metrics(points: list[Mapping[str, Any]], *, per_version: bool) -> None:
    print("Flux MoE replay metrics:")
    for point in points:
        if point.get("status") != "completed":
            continue
        label = (
            f"v{int(point['version_number']):03d}"
            if per_version
            else str(point.get("version"))
        )
        fields = " ".join(f"{key}={point.get(key)}" for key in METRIC_KEYS)
        print(f"  {label} {fields} valid={point.get('valid')}")


def _wrap_text_summary(name: str, *, per_version: bool) -> None:
    original = getattr(cli, name)

    def command(args: Any) -> None:
        original(args)
        if args.dry_run or args.json:
            return
        log_dir = Path(os.environ["SFORGE_LOG_DIR"])
        if per_version:
            path = (
                log_dir
                / "runs"
                / args.replay_run_id
                / TASK_ID
                / "version-replays"
                / args.run_id
                / args.measurement_set
                / "versions_curve.json"
            )
        else:
            path = (
                log_dir
                / "runs"
                / args.replay_run_id
                / TASK_ID
                / "candidate-replays"
                / args.run_id
                / args.measurement_set
                / "curve.json"
            )
        curve = json.loads(path.read_text(encoding="utf-8"))
        _print_metrics(curve["points"], per_version=per_version)

    setattr(cli, name, command)


candidate_replay.rewrite_selector_archive = _minimal_version_archive
_install_rich_candidate_curve()
_install_rich_version_curve()
_wrap_text_summary("cmd_replay_candidates", per_version=False)
_wrap_text_summary("cmd_replay_versions", per_version=True)


if __name__ == "__main__":
    cli.main()
