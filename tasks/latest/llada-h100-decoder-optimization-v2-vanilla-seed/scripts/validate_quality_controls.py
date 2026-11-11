#!/usr/bin/env python3
"""Validate the precommitted independent positive/negative control manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from h100_policy_contract import PROFILE, TASK_ID, is_sha256, object_at, sha256


def validate_control(item: object, expected: str, *, root: Path, verify_hash: bool) -> None:
    if not isinstance(item, dict):
        raise ValueError("quality control entry is not an object")
    if (
        re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", str(item.get("id", ""))
        )
        is None
        or item.get("expected_classification") != expected
        or not isinstance(item.get("archive"), str)
        or not is_sha256(item.get("archive_sha256"))
        or not isinstance(item.get("acceptance_target"), dict)
    ):
        raise ValueError("quality control entry is incomplete")
    if verify_hash:
        path = (root / item["archive"]).resolve()
        if root not in path.parents or path.is_symlink() or not path.is_file():
            raise ValueError("quality control archive path is unsafe or missing")
        if sha256(path) != item["archive_sha256"]:
            raise ValueError("quality control archive hash mismatch")


def main() -> int:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, default=default_root)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--verify-archive-hashes", action="store_true")
    parser.add_argument("--require-precommitted", action="store_true")
    args = parser.parse_args()
    root = args.task_root.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else root / "quality_controls.json"
    manifest = object_at(manifest_path, "quality-control manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("task_id") != TASK_ID
        or manifest.get("profile") != PROFILE
        or manifest.get("historical_a6000_controls_applicability") != "NONE"
    ):
        raise SystemExit("quality-control manifest identity is invalid")
    methodology = manifest.get("methodology")
    if (
        not isinstance(methodology, dict)
        or methodology.get("formula")
        != "producer_relative_min_retention_ppm_v1"
        or methodology.get("metric")
        != "producer_relative_min_retention_ppm_v1"
        or methodology.get("threshold_derivation")
        != "midpoint_min_positive_max_negative_v1"
        or methodology.get("margin_selection")
        != "producer_derived_after_precommitted_controls"
        or isinstance(methodology.get("minimum_separation_ppm"), bool)
        or not isinstance(methodology.get("minimum_separation_ppm"), int)
        or methodology["minimum_separation_ppm"] <= 0
        or isinstance(methodology.get("control_measurement_floor_ppm"), bool)
        or not isinstance(methodology.get("control_measurement_floor_ppm"), int)
        or methodology["control_measurement_floor_ppm"] <= 0
    ):
        raise SystemExit("quality-control threshold methodology is invalid")
    status = manifest.get("status")
    if status == "PENDING_CONTROL_ARCHIVES" and not args.require_precommitted:
        if manifest.get("positive_controls") or manifest.get("negative_controls"):
            raise SystemExit("pending quality-control manifest contains partial controls")
        print("PENDING_CONTROL_ARCHIVES")
        return 0
    if status != "PRECOMMITTED":
        raise SystemExit("quality controls are not precommitted")
    positives = manifest.get("positive_controls")
    negatives = manifest.get("negative_controls")
    if not isinstance(positives, list) or not positives or not isinstance(negatives, list) or not negatives:
        raise SystemExit("both positive and negative independent controls are required")
    ids = set()
    for expected, controls in (("ACCEPT", positives), ("REJECT", negatives)):
        for item in controls:
            validate_control(item, expected, root=root, verify_hash=args.verify_archive_hashes)
            if item["id"] in ids:
                raise SystemExit("quality control IDs must be unique")
            ids.add(item["id"])
    if not is_sha256(manifest.get("workload_policy_sha256")) or not isinstance(
        manifest.get("approval_id"), str
    ) or not manifest["approval_id"]:
        raise SystemExit("precommitted quality-control provenance is incomplete")
    calibration = manifest.get("producer_calibration")
    if (
        not isinstance(calibration, dict)
        or calibration.get("expected_mode") != "calibration"
        or not isinstance(calibration.get("archive"), str)
        or not is_sha256(calibration.get("archive_sha256"))
    ):
        raise SystemExit("producer calibration archive is not precommitted")
    if args.verify_archive_hashes:
        calibration_path = (root / calibration["archive"]).resolve()
        if (
            root not in calibration_path.parents
            or calibration_path.is_symlink()
            or not calibration_path.is_file()
            or sha256(calibration_path) != calibration["archive_sha256"]
        ):
            raise SystemExit("producer calibration archive hash mismatch")
    print(sha256(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
