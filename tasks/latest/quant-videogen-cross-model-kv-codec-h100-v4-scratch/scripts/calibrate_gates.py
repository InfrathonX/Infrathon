#!/usr/bin/env python3
"""Calibrate the v4 release from trusted-oracle Judge runs (private qvg-private.json files).

Rule: generation is not bit-stable on every model, and VBench is not either.
HY frames reproduce bit-for-bit (hy-qvg matches the v3 golden a8ee4657 on another
host), but LongCat frames drift slightly between runs (PSNR <= 0.09 dB), and
VBench scores the same frames differently from run to run (per-case sigma about
0.2-0.35 points on HY, 0.4-0.6 on LongCat; the LongCat 6-case mean moved up to
0.65 points between the two 2026-10-07 oracle runs). So calibrate from several
runs, take the worst reading over runs (the 3-sigma term is kept for more runs),
and give VBench means a 1.0-point margin.

* Mean gates (per model, on the case means): PSNR - 0.25 dB, SSIM - 0.0075,
  LPIPS + 0.0075; VBench allowed drop = observed drop below the mean BF16 VBench
  (at least 0) + 1.0 point (run-to-run noise, see above).
* Per-case floors (per model, from the worst case): PSNR - 1.0 dB, SSIM - 0.03,
  LPIPS + 0.03; VBench allowed drop = 2 x (worst per-case drop, at least 0, + 0.25).
* Each case's bf16_kv_bytes is frozen from the codec-mode oracle profile (shapes only).

Usage: calibrate_gates.py PRIVATE_JSON [PRIVATE_JSON ...] [--write]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def release_path(task_root: Path) -> Path:
    """The Judge-only release config: the in-repository copy when present, else the
    asset-root copy ``<INFRATHON_ASSETS_ROOT>/<task directory>/judge/h100_scratch_v4_release.json``
    written by ``assets/fetch.sh`` (default root ``~/.cache/infrathon/assets``)."""
    local = task_root / "configs/h100_scratch_v4_release.json"
    if local.is_file():
        return local
    assets = os.environ.get("INFRATHON_ASSETS_ROOT") or str(
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "infrathon" / "assets")
    return Path(assets).expanduser() / task_root.name / "judge/h100_scratch_v4_release.json"


RELEASE = release_path(ROOT)
DIMENSIONS = ("aesthetic_quality", "background_consistency", "image_quality", "subject_consistency")
MEAN_MARGIN = {"psnr": 0.25, "ssim": 0.0075, "lpips": 0.0075, "vbench": 1.0}
FLOOR_MARGIN = {"psnr": 1.0, "ssim": 0.03, "lpips": 0.03, "vbench": 0.25}


def low(values, margin):
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    return min(min(values) - margin, statistics.fmean(values) - 3 * sd)


def high(values, margin):
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    return max(max(values) + margin, statistics.fmean(values) + 3 * sd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    release = json.loads(RELEASE.read_text())
    runs = [json.loads(path.read_text()) for path in args.runs]
    if len(runs) < 2:
        raise SystemExit("calibration needs at least two oracle runs (VBench is not run-to-run stable)")
    for run in runs:
        calibrating = run.get("release_status") == "calibrating" and run.get("reason") == "uncalibrated_release"
        calibrated = run.get("release_status") == "calibrated" and not run.get("infrastructure_invalid")
        if run.get("outcome") != "evaluated" or not (calibrating or calibrated):
            raise SystemExit("calibration needs complete oracle runs of this release")
    by_case = {case["case_id"]: case for case in release["cases"]}
    rows = {}  # case_id -> list of rows over runs
    for run in runs:
        cases = run["cases"] if isinstance(run["cases"], list) else list(run["cases"].values())
        for case in cases:
            rows.setdefault(case["control"]["case_id"], []).append(case["row"])
    if set(rows) != set(by_case) or any(len(v) != len(runs) for v in rows.values()):
        raise SystemExit("every run must cover every release case")
    report = {}
    for model, spec in release["models"].items():
        members = [cid for cid, case in by_case.items() if case["model"] == model]
        mean = {"psnr": [], "ssim": [], "lpips": [], **{d: [] for d in DIMENSIONS}}
        for run_index in range(len(runs)):
            run_rows = [rows[cid][run_index] for cid in members]
            mean["psnr"].append(statistics.fmean(r["fidelity"]["psnr"] for r in run_rows))
            mean["ssim"].append(statistics.fmean(r["fidelity"]["ssim"] for r in run_rows))
            mean["lpips"].append(statistics.fmean(r["fidelity"]["lpips"] for r in run_rows))
            for d in DIMENSIONS:
                bf16 = statistics.fmean(by_case[cid]["bf16_vbench"][d] for cid in members)
                mean[d].append(bf16 - statistics.fmean(r["vbench"][d] for r in run_rows))  # drop
        case_psnr = [r["fidelity"]["psnr"] for cid in members for r in rows[cid]]
        case_ssim = [r["fidelity"]["ssim"] for cid in members for r in rows[cid]]
        case_lpips = [r["fidelity"]["lpips"] for cid in members for r in rows[cid]]
        case_drop = {d: [by_case[cid]["bf16_vbench"][d] - r["vbench"][d] for cid in members for r in rows[cid]]
                     for d in DIMENSIONS}
        spec["gates"] = {
            "mean": {
                "psnr_min": low(mean["psnr"], MEAN_MARGIN["psnr"]),
                "ssim_min": low(mean["ssim"], MEAN_MARGIN["ssim"]),
                "lpips_max": high(mean["lpips"], MEAN_MARGIN["lpips"]),
                "vbench_allowed_drop": {d: max(0.0, high(mean[d], 0.0)) + MEAN_MARGIN["vbench"] for d in DIMENSIONS},
            },
            "case_floor": {
                "psnr_min": min(case_psnr) - FLOOR_MARGIN["psnr"],
                "ssim_min": min(case_ssim) - FLOOR_MARGIN["ssim"],
                "lpips_max": max(case_lpips) + FLOOR_MARGIN["lpips"],
                "vbench_allowed_drop": {d: 2 * (max(0.0, max(case_drop[d])) + FLOOR_MARGIN["vbench"])
                                        for d in DIMENSIONS},
            },
        }
        report[model] = spec["gates"]
    for cid, case in by_case.items():
        values = {r["bf16_kv_bytes"] for r in rows[cid]}
        if len(values) != 1:
            raise SystemExit(f"{cid}: bf16_kv_bytes differs between runs: {sorted(values)}")
        case["bf16_kv_bytes"] = values.pop()
    release["status"] = "calibrated"
    print(json.dumps(report, indent=1))
    if args.write:
        RELEASE.write_text(json.dumps(release, indent=2) + "\n")
        print(f"wrote {RELEASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
