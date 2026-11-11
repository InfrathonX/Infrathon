#!/usr/bin/env python3
"""CPU-only layout validation for the Quant-VideoGen scratch v4 task.

Checks the release file set, the task JSON bindings, the schema-2 release (17
hidden cases, the sharded-Judge block, per-model budgets and gates, and the
rule that a digest or a calibrated value may be null only while the release is
still ``calibrating``), the self-contained Work base (a digest-pinned public
image plus a fully pinned requirement lock, built from this repository before
the Work image), and the local Judge-only path: ``scripts/eval_archive.sh``
locks every Judge GPU under both host lock names and hands the whole UUID list
to the adapter, while the controller plans its cases with
``judge/judge_shards.py``.  The v3 single-GPU Work+Judge path must be gone.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

TASK_ID = "quant_videogen_cross_model_kv_codec_h100_scratch_v4"
RELEASE = "configs/h100_scratch_v4_release.json"
QUALITY = "configs/h100_scratch_v4_quality.json"
QUALITY_KEYS = {"schema_version", "hardware", "models", "oracle_calibration"}
PROMPTS = "judge/private/prompts/MovieGenVideoBench_extended.txt"
REQUIRED = (
    "README.md",
    "images/work/Dockerfile", "images/work/AGENT_TASK.md", "images/work/codex-config.toml",
    "images/work/base/Dockerfile", "images/work/base/requirements.lock",
    "images/work/base/check_environment.py",
    "images/work/public_tests/run_public_tests.py", "images/work/public_tests/run_self_forcing_paper.py",
    "images/judge/Dockerfile",
    "judge/run_judge.py", "judge/evaluate_hidden.py", "judge/judge_shards.py", "judge/run_longcat.py",
    "judge/run_hy_worldplay_8b.py", PROMPTS, "scripts/stage_eval_inputs.py",
    RELEASE, "configs/paper_protocol.json",
    "metrics/compare_frames.py", "metrics/run_vbench.py", "policy/codec_source_policy.py",
    "runtime/quant_videogen/payload.py", "assets/public_prompts.txt", "locks/model-lock.json",
    "oracle/mlsysforge_codec.py",
    "scripts/lib.sh", "scripts/eval_archive.sh", "scripts/refresh_portable_manifests.py", "scripts/sforge_runtime_entry.py",
    "scripts/build_images.sh", "scripts/run_oracle.sh", "scripts/replay_candidates.sh", "scripts/validate_layout.py",
    "portable/REBUILD_PROMPT.md",
    "portable/resource-manifest.json", "portable/build-resource-manifest.json", "portable/public-suite-manifest.json",
    "portable/interactive-suite-manifest.json", "portable/final-suite-manifest.json", "portable/oracle-protocol.json",
    "portable/selection-protocol.json", "portable/software-abi.json", "portable/calibration-protocol.json",
    "tests/test_portable_contract.py", "tests/test_runner_contract.py", "tests/test_shards.py",
    "tests/test_judge_isolation.py", "tests/test_eval_archive.py",
    f"tasks/{TASK_ID}.json",
)
# The v3 local Work+Judge single-GPU path.  Agent runs happen on PyroMind; the
# local path is Judge-only (scripts/eval_archive.sh), as in kvpress v2.
REMOVED = (
    "scripts/run_fixed_budget.sh", "scripts/run_single_gpu_fixed_budget.sh",
    "scripts/single_gpu_runtime_preflight.py",
    "scripts/run_qvg_public_vbench_case.sh", "scripts/calibrate_self_vbench.py",
    "docs/SINGLE_GPU_MIGRATION.md", "tests/test_single_gpu_migration.py",
)
# Per-model fixed-budget wrappers (scripts/run_<model>_<effort>_<N>h.sh) and
# their experiment-<model>-<effort>-<N>h.yaml configs went with the local path.
REMOVED_GLOBS = ("scripts/launch_*.sh", "scripts/run_model_*.sh", "scripts/run_*_[0-9]*h.sh", "experiment-*.yaml")

MODELS = ("longcat", "hy_worldplay_8b")
LONGCAT_CASES = ("lc-012", "lc-185", "lc-320", "lc-593", "lc-893", "lc-894")
HY_CASES = (*(f"hy-{n:02d}" for n in range(1, 11)), "hy-qvg")
PUBLIC_PROMPT_RECORDS = (1, 6)  # one-based; paper_protocol public_prompt_records [0, 5]
VBENCH = ("aesthetic_quality", "background_consistency", "image_quality", "subject_consistency")
SHA256 = re.compile(r"[0-9a-f]{64}")
HIDDEN = "/eval/hidden"
TOP_KEYS = {"status", "protocol_schema_version", "paper_protocol_sha256", "metric_implementation_sha256",
            "prompt_file", "prompt_file_sha256", "judge", "models", "cases"}
JUDGE_KEYS = {"max_gpu_count", "min_gpu_count", "slot_uids", "master_port_base", "threads_per_slot",
              "start_stagger_seconds", "case_overhead_seconds", "controller_overhead_seconds",
              "min_device_memory_bytes"}
MODEL_KEYS = {"fps", "stage_seconds", "fidelity_metric_seconds", "vbench_metric_seconds", "profile_key",
              "frames_file", "gates"}
MODEL_OUTPUTS = {
    "longcat": {"fps": 15, "profile_key": "persistent_kv", "frames_file": "final.frames.pt"},
    "hy_worldplay_8b": {"fps": 24, "profile_key": "persistent_vision_kv", "frames_file": "gen.frames.pt"},
}
GATE_KEYS = {"psnr_min", "ssim_min", "lpips_max", "vbench_allowed_drop"}
COMMON_CASE_KEYS = {"case_id", "model", "seed", "prompt", "bf16_frames", "bf16_frames_sha256", "frame_count",
                    "fidelity_frames", "bf16_vbench", "bf16_kv_bytes"}
CASE_KEYS = {
    "longcat": COMMON_CASE_KEYS | {"prompt_index", "segments", "init_video", "init_video_sha256"},
    "hy_worldplay_8b": COMMON_CASE_KEYS | {"image", "image_sha256", "image_source", "video_length", "pose"},
}
LONGCAT_CONTEXT_FRAMES, LONGCAT_SEGMENT_FRAMES = 73, 20
HY_SOURCE = re.compile(r"HY-WorldPlay (assets/img/[0-9a-z_]+\.(png|jpeg)) @1588e133")


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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_int(value: Any, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float("inf")


def _gate_errors(name: str, gates: Any, calibrated: bool) -> list[str]:
    if not isinstance(gates, dict) or set(gates) != GATE_KEYS:
        return [f"{name}: gate keys must be {sorted(GATE_KEYS)}"]
    errors = [f"{name}.{key} is not a finite number" for key in ("psnr_min", "ssim_min", "lpips_max")
              if not _is_number(gates[key])]
    drops = gates["vbench_allowed_drop"]
    if not isinstance(drops, dict) or set(drops) != set(VBENCH) or not all(_is_number(v) and v >= 0 for v in drops.values()):
        errors.append(f"{name}.vbench_allowed_drop must give a non-negative drop for {list(VBENCH)}")
    if errors or not calibrated:
        return errors
    # Placeholders (0 / 1 / 100) admit everything; a calibrated release must not keep them.
    if not (gates["psnr_min"] > 0 and gates["ssim_min"] > 0 and gates["lpips_max"] < 1
            and all(v < 100 for v in drops.values())):
        errors.append(f"{name}: calibrated release still carries placeholder gates")
    return errors


def release_errors(release: dict[str, Any], root: Path) -> list[str]:
    """Every schema-2 rule the controller and the manifests rely on."""
    errors: list[str] = []
    missing = TOP_KEYS - set(release)
    if missing:
        return [f"release lacks {sorted(missing)}"]
    status = release["status"]
    if status not in ("calibrating", "calibrated"):
        errors.append(f"release status must be calibrating or calibrated, not {status!r}")
    calibrated = status == "calibrated"
    if release["protocol_schema_version"] != 2:
        errors.append("release protocol_schema_version must be 2")
    pins = {
        "configs/paper_protocol.json": release["paper_protocol_sha256"],
        "metrics/compare_frames.py": release["metric_implementation_sha256"].get("fidelity"),
        "metrics/run_vbench.py": release["metric_implementation_sha256"].get("vbench"),
        PROMPTS: release["prompt_file_sha256"],
    }
    for relative, expected in pins.items():
        if _sha(root / relative) != expected:
            errors.append(f"{relative} does not match the release")
    if release["prompt_file"] != f"{HIDDEN}/prompts/MovieGenVideoBench_extended.txt":
        errors.append("release prompt_file must be the hidden MovieGen prompt file")

    judge = release["judge"]
    if not isinstance(judge, dict) or set(judge) != JUDGE_KEYS:
        errors.append(f"release judge keys must be {sorted(JUDGE_KEYS)}")
    else:
        counts_ok = _is_int(judge["min_gpu_count"], 1) and _is_int(judge["max_gpu_count"], 1)
        if not counts_ok or judge["min_gpu_count"] > judge["max_gpu_count"]:
            errors.append("judge GPU counts must satisfy 1 <= min_gpu_count <= max_gpu_count")
        uids = judge["slot_uids"]
        if (not isinstance(uids, list) or not all(_is_int(uid, 1000) for uid in uids) or len(set(uids)) != len(uids)
                or (counts_ok and len(uids) != judge["max_gpu_count"])):
            errors.append("judge.slot_uids must be max_gpu_count distinct unprivileged UIDs")
        if not _is_int(judge["master_port_base"], 1024) or judge["master_port_base"] + len(release["cases"]) > 65535:
            errors.append("judge.master_port_base leaves no unprivileged port per case")
        for key in ("threads_per_slot", "min_device_memory_bytes", "controller_overhead_seconds"):
            if not _is_int(judge[key], 1):
                errors.append(f"judge.{key} must be a positive integer")
        for key in ("start_stagger_seconds", "case_overhead_seconds"):
            if not _is_int(judge[key], 0):
                errors.append(f"judge.{key} must be a non-negative integer")

    models = release["models"]
    if not isinstance(models, dict) or set(models) != set(MODELS):
        errors.append(f"release models must be exactly {list(MODELS)}")
        models = {}
    for name, model in models.items():
        if not isinstance(model, dict) or set(model) != MODEL_KEYS:
            errors.append(f"models.{name} keys must be exactly {sorted(MODEL_KEYS)} "
                          "(the quality reference is per device class, in the quality file)")
            continue
        for key, expected in MODEL_OUTPUTS[name].items():
            if model[key] != expected:
                errors.append(f"models.{name}.{key} must be {expected!r}")
        for key in ("stage_seconds", "fidelity_metric_seconds", "vbench_metric_seconds"):
            if not _is_int(model[key], 1):
                errors.append(f"models.{name}.{key} must be a positive integer")
        gates = model["gates"]
        if not isinstance(gates, dict) or set(gates) != {"mean", "case_floor"}:
            errors.append(f"models.{name}.gates must hold mean and case_floor")
            continue
        gate_errors = _gate_errors(f"models.{name}.gates.mean", gates["mean"], calibrated)
        gate_errors += _gate_errors(f"models.{name}.gates.case_floor", gates["case_floor"], calibrated)
        errors += gate_errors
        if not gate_errors:
            mean, floor = gates["mean"], gates["case_floor"]
            if (floor["psnr_min"] > mean["psnr_min"] or floor["ssim_min"] > mean["ssim_min"]
                    or floor["lpips_max"] < mean["lpips_max"]
                    or any(floor["vbench_allowed_drop"][d] < mean["vbench_allowed_drop"][d] for d in VBENCH)):
                errors.append(f"models.{name}: the per-case floor must be no stricter than the mean gate")

    errors += _case_errors(release, root, calibrated)
    return errors


def quality_paths(task_root: Path) -> list[Path]:
    """The device-class quality files present: the in-repository copy when present, else
    every ``<asset root>/<task directory>/hw/<type>/judge/h100_scratch_v4_quality.json``
    (a host fetches only its own type's)."""
    local = task_root / QUALITY
    if local.is_file():
        return [local]
    return sorted(release_path(task_root).parents[1].glob(f"hw/*/judge/{Path(QUALITY).name}"))


def validate_quality(root: Path, errors: list[str]) -> None:
    """Each device class's quality file: hardware type, one reference per model, oracle band."""
    release = json.loads(release_path(root).read_text(encoding="utf-8"))
    paths = quality_paths(root)
    if release.get("status") == "calibrated" and not paths:
        errors.append(f"a calibrated release needs this device class's {Path(QUALITY).name} (run assets/fetch.sh)")
    for path in paths:
        label = path.relative_to(release_path(root).parents[1]) if path != root / QUALITY else Path(QUALITY)
        try:
            quality = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{label}: unreadable: {exc}")
            continue
        if not isinstance(quality, dict) or set(quality) != QUALITY_KEYS or quality.get("schema_version") != 1:
            errors.append(f"{label}: must be schema 1 with exactly {sorted(QUALITY_KEYS)}")
            continue
        if path != root / QUALITY and quality["hardware"] != path.parents[1].name:
            errors.append(f"{label}: hardware {quality['hardware']!r} differs from its hw/ directory")
        models = quality["models"]
        if not isinstance(models, dict) or set(models) != set(MODELS):
            errors.append(f"{label}: models must be exactly {list(MODELS)}")
            continue
        for name, entry in models.items():
            if not isinstance(entry, dict) or set(entry) != {"quality_reference"}:
                errors.append(f"{label}: models.{name} must hold exactly quality_reference")
                continue
            mean_gate = (release.get("models", {}).get(name, {}).get("gates") or {}).get("mean")
            errors += [f"{label}: {error}" for error in _quality_reference_errors(name, entry["quality_reference"], mean_gate)]
        band = quality["oracle_calibration"].get("score_band") if isinstance(quality["oracle_calibration"], dict) else None
        if not (isinstance(band, list) and len(band) == 2 and all(_is_number(x) for x in band) and 0 < band[0] < band[1]):
            errors.append(f"{label}: oracle_calibration.score_band must be [low, high] with 0 < low < high")


def _quality_reference_errors(name: str, reference: Any, mean_gate: Any) -> list[str]:
    """The zero point of q: the reference codec's case means, device-measured like the gates."""
    context = f"models.{name}.quality_reference"
    if not isinstance(reference, dict) or set(reference) != {"psnr", "ssim", "lpips", "vbench"}:
        return [f"{context} must hold exactly psnr, ssim, lpips and vbench"]
    vbench = reference["vbench"]
    if not isinstance(vbench, dict) or set(vbench) != set(VBENCH):
        return [f"{context}.vbench must name exactly the VBench dimensions"]
    numbers = [reference[k] for k in ("psnr", "ssim", "lpips")] + [vbench[d] for d in VBENCH]
    if not all(_is_number(x) for x in numbers):
        return [f"{context} values must be finite numbers"]
    if not (reference["psnr"] < 40.0 and 0.0 < reference["ssim"] < 1.0 and reference["lpips"] > 0.0):
        return [f"{context} must lie strictly inside the fidelity ideals"]
    # The reference codec is admitted: its case means clear the mean fidelity gates.
    if isinstance(mean_gate, dict) and all(_is_number(mean_gate.get(k)) for k in ("psnr_min", "ssim_min", "lpips_max")):
        if (reference["psnr"] < mean_gate["psnr_min"] or reference["ssim"] < mean_gate["ssim_min"]
                or reference["lpips"] > mean_gate["lpips_max"]):
            return [f"{context} fails its own mean fidelity gate"]
    return []


def _case_errors(release: dict[str, Any], root: Path, calibrated: bool) -> list[str]:
    errors: list[str] = []
    cases = release["cases"]
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        return ["release cases must be a list of objects"]
    ids = [case.get("case_id") for case in cases]
    if sorted(map(str, ids)) != sorted((*LONGCAT_CASES, *HY_CASES)):
        errors.append(f"release cases must be exactly {[*LONGCAT_CASES, *HY_CASES]}")
    prompts = (root / PROMPTS).read_text(encoding="utf-8").splitlines()
    paper = json.loads((root / "configs/paper_protocol.json").read_text(encoding="utf-8"))["mlsysforge"]
    windows = paper["fidelity_windows"]
    suite = paper["hidden_suite"]
    lock = json.loads((root / "locks/model-lock.json").read_text(encoding="utf-8"))
    locked_images = {entry["path"]: entry["sha256"]
                     for entry in lock["git_bundles"]["hy_worldplay_official_assets"]["files"]}
    paths: list[str] = []

    def nullable_sha(case_id: str, key: str, value: Any) -> None:
        if value is None:
            if calibrated:
                errors.append(f"{case_id}.{key} is null in a calibrated release")
        elif not isinstance(value, str) or not SHA256.fullmatch(value):
            errors.append(f"{case_id}.{key} must be a sha256 hex digest or null")

    for case in cases:
        case_id, model = case.get("case_id"), case.get("model")
        if model not in CASE_KEYS or set(case) != CASE_KEYS[model]:
            errors.append(f"{case_id}: keys must be {sorted(CASE_KEYS.get(model, COMMON_CASE_KEYS))}")
            continue
        if not _is_int(case["seed"]) or not isinstance(case["prompt"], str) or not case["prompt"].strip():
            errors.append(f"{case_id}: seed and prompt are required")
        nullable_sha(case_id, "bf16_frames_sha256", case["bf16_frames_sha256"])
        kv = case["bf16_kv_bytes"]
        if kv is None:
            if calibrated:
                errors.append(f"{case_id}.bf16_kv_bytes is null in a calibrated release")
        elif not _is_int(kv, 1):
            errors.append(f"{case_id}.bf16_kv_bytes must be a positive integer or null")
        vbench = case["bf16_vbench"]
        if not isinstance(vbench, dict) or set(vbench) != set(VBENCH) or not all(_is_number(v) and v >= 0 for v in vbench.values()):
            errors.append(f"{case_id}.bf16_vbench must give a non-negative reading for {list(VBENCH)}")
        elif calibrated and not all(v > 0 for v in vbench.values()):
            errors.append(f"{case_id}.bf16_vbench still holds placeholders in a calibrated release")
        frames = case["fidelity_frames"]
        if frames != windows[model]:
            errors.append(f"{case_id}.fidelity_frames must be the paper window {windows[model]}")
        if not _is_int(case["frame_count"], 1) or not (isinstance(frames, list) and len(frames) == 2
                                                       and 0 <= frames[0] < frames[1] <= case["frame_count"]):
            errors.append(f"{case_id}: fidelity_frames must lie inside frame_count")
        paths.append(case["bf16_frames"])
        if model == "longcat":
            index = case["prompt_index"]
            if not _is_int(index, 1) or index > len(prompts) or case_id != f"lc-{index:03d}":
                errors.append(f"{case_id}: prompt_index must be the one-based record its id names")
                continue
            if index in PUBLIC_PROMPT_RECORDS:
                errors.append(f"{case_id}: prompt record {index} is public")
            if case["prompt"] != prompts[index - 1]:
                errors.append(f"{case_id}: prompt is not record {index} of the hidden prompt file")
            segments = case["segments"]
            if segments != suite["longcat"]["segments"]:
                errors.append(f"{case_id}: segments must be {suite['longcat']['segments']}")
            elif case["frame_count"] != LONGCAT_CONTEXT_FRAMES + LONGCAT_SEGMENT_FRAMES * segments:
                errors.append(f"{case_id}: frame_count must be 73 + 20 x segments")
            base = f"{HIDDEN}/longcat/{index:03d}"
            if case["init_video"] != f"{base}/init.mp4" or case["bf16_frames"] != f"{base}/bf16-{segments}seg.frames.pt":
                errors.append(f"{case_id}: hidden paths must be {base}/init.mp4 and {base}/bf16-{segments}seg.frames.pt")
            nullable_sha(case_id, "init_video_sha256", case["init_video_sha256"])
            paths.append(case["init_video"])
        else:
            name = case_id.removeprefix("hy-")
            source = HY_SOURCE.fullmatch(str(case["image_source"]))
            if source is None:
                errors.append(f"{case_id}: image_source must name an assets/img file of HY-WorldPlay @1588e133")
                continue
            base = f"{HIDDEN}/hy/{name}"
            if case["image"] != f"{base}/image.{source.group(2)}":
                errors.append(f"{case_id}: image must be {base}/image.{source.group(2)}")
            if locked_images.get(source.group(1)) != case["image_sha256"]:
                errors.append(f"{case_id}: image_sha256 differs from the model lock's {source.group(1)}")
            length = case["video_length"]
            if not _is_int(length, 5) or (length - 1) % 4 or case["frame_count"] != length:
                errors.append(f"{case_id}: video_length must be 4n+1 and equal frame_count")
            elif case["bf16_frames"] != f"{base}/bf16-{length}.frames.pt":
                errors.append(f"{case_id}: bf16_frames must be {base}/bf16-{length}.frames.pt")
            if case["pose"] != suite["hy_worldplay_8b"]["pose"]:
                errors.append(f"{case_id}: pose must be the paper trajectory {suite['hy_worldplay_8b']['pose']}")
            elif _is_int(length, 5) and 4 * sum(int(step.rsplit("-", 1)[1]) for step in case["pose"].split(",")) + 1 != length:
                errors.append(f"{case_id}: pose does not cover video_length")
            paths.append(case["image"])
    if len(set(paths)) != len(paths):
        errors.append("release cases share a hidden input or golden path")
    longcat = sorted(case.get("prompt_index") for case in cases if case.get("model") == "longcat")
    if longcat != sorted(suite["longcat"]["prompt_records_one_based"]):
        errors.append("LongCat prompt records differ from configs/paper_protocol.json")
    if sum(case.get("model") == "hy_worldplay_8b" for case in cases) != suite["hy_worldplay_8b"]["cases"]:
        errors.append("HY case count differs from configs/paper_protocol.json")
    return errors


def validate_release(root: Path, errors: list[str]) -> None:
    """The release pins the shipped protocol, metrics, prompts and case table."""
    release = json.loads(release_path(root).read_text(encoding="utf-8"))
    errors += release_errors(release, root)
    hyworld = root / "assets/hyworld.png"
    qvg = [case for case in release["cases"] if case.get("case_id") == "hy-qvg"]
    if hyworld.is_file() and qvg and qvg[0].get("image_sha256") != _sha(hyworld):
        errors.append("hy-qvg must use the v3 HY image (assets/hyworld.png)")


def validate_private_inputs(root: Path, errors: list[str]) -> None:
    """judge/private/ is a Judge-only input tree: mounted at run time, never baked."""
    ignore = (root / ".dockerignore").read_text(encoding="utf-8").splitlines()
    if "judge/private/" not in [line.strip() for line in ignore]:
        errors.append(".dockerignore must exclude judge/private/ from every build context")
    for dockerfile in sorted(root.glob("images/**/Dockerfile")):
        if "judge/private" in dockerfile.read_text(encoding="utf-8"):
            errors.append(f"{dockerfile.relative_to(root)} references judge/private")
    prompts = root / PROMPTS
    if prompts.is_file() and len(prompts.read_text(encoding="utf-8").splitlines()) != 1003:
        errors.append(f"{PROMPTS} must hold the 1003 Movie Gen Video Bench prompts")
    # Case images come from the hidden tree (/eval/hidden/hy/...), never the image.
    judge = (root / "images/judge/Dockerfile").read_text(encoding="utf-8")
    if "hyworld.png" in judge or "qvg-judge-assets" in judge:
        errors.append("images/judge/Dockerfile must not bake HY case images")


BASE_IMAGE_TAG = (
    "quant_videogen_h100_scratch_v4.base."
    "quant_videogen_cross_model_kv_codec_h100_scratch_v4:local"
)


def validate_self_contained_base(root: Path, errors: list[str]) -> None:
    """The Work base must be buildable from this repository alone."""
    work = (root / "images/work/Dockerfile").read_text(encoding="utf-8")
    if f"ARG BASE_WORK_IMAGE={BASE_IMAGE_TAG}" not in work:
        errors.append("Work Dockerfile must default to the task-owned Work base image")
    if "quant_videogen_h100_v2.work" in work:
        errors.append("Work Dockerfile must not default to an out-of-tree v2 image")
    if "portable" in work.lower() or "final-suite-manifest" in work.lower():
        errors.append("Work Dockerfile must not reference the host-private portable manifests")
    base = (root / "images/work/base/Dockerfile").read_text(encoding="utf-8")
    from_lines = [line for line in base.splitlines() if line.startswith("ARG BASE_IMAGE=")]
    if len(from_lines) != 1 or "@sha256:" not in from_lines[0]:
        errors.append("Work base must start from a single digest-pinned public image")
    if not any(line.startswith("FROM ${BASE_IMAGE}") for line in base.splitlines()):
        errors.append("Work base must build FROM the pinned BASE_IMAGE argument")
    for forbidden in ("nodejs.org", "@openai/codex", "npm install"):
        if forbidden in base:
            errors.append(f"Work base must stay agent-free: found {forbidden!r}")
    lock = root / "images/work/base/requirements.lock"
    for number, line in enumerate(lock.read_text(encoding="utf-8").splitlines(), start=1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if " @ " in entry:
            if "#sha256=" not in entry:
                errors.append(f"requirements.lock:{number}: URL requirement lacks a sha256")
        elif "==" not in entry:
            errors.append(f"requirements.lock:{number}: requirement is not pinned: {entry}")
    build = (root / "scripts/build_images.sh").read_text(encoding="utf-8")
    base_build = build.find("images/work/base/Dockerfile")
    work_build = build.find("images/work/Dockerfile")
    judge_build = build.find("images/judge/Dockerfile")
    if not 0 <= base_build < work_build < judge_build:
        errors.append("scripts/build_images.sh must build base, then Work, then Judge")
    if '"${BASE_WORK_IMAGE}" == "${DEFAULT_BASE_WORK_IMAGE}"' not in build:
        errors.append("scripts/build_images.sh must keep the external-base override path")


def validate_sharded_judge(root: Path, errors: list[str]) -> None:
    """Judge-only local path over 1-8 locked GPUs; no single-GPU Work+Judge path."""
    present = [name for name in REMOVED if (root / name).exists()]
    present += [path.relative_to(root).as_posix() for pattern in REMOVED_GLOBS for path in sorted(root.glob(pattern))]
    if present:
        errors.append("the v3 single-GPU Work+Judge path is still present: " + ", ".join(present))
    lib = (root / "scripts/lib.sh").read_text(encoding="utf-8")
    for fragment in ("/sforge-gpu-", "/mlsysforge-gpu-", "flock"):
        if fragment not in lib:
            errors.append(f"scripts/lib.sh: missing {fragment!r} (both host GPU lock names)")
    launcher = (root / "scripts/eval_archive.sh").read_text(encoding="utf-8")
    for fragment in ("JUDGE_GPUS", "QVG_JUDGE_GPU_UUIDS", "lock_gpu", " eval"):
        if fragment not in launcher:
            errors.append(f"scripts/eval_archive.sh: missing {fragment!r}")
    if "QVG_JUDGE_GPU_UUIDS" not in (root / "scripts/sforge_runtime_entry.py").read_text(encoding="utf-8"):
        errors.append("scripts/sforge_runtime_entry.py must take the Judge GPUs from QVG_JUDGE_GPU_UUIDS")
    if "eval_archive.sh" not in (root / "scripts/run_oracle.sh").read_text(encoding="utf-8"):
        errors.append("scripts/run_oracle.sh must evaluate through scripts/eval_archive.sh")
    if "def plan_shards(" not in (root / "judge/judge_shards.py").read_text(encoding="utf-8"):
        errors.append("judge/judge_shards.py must define plan_shards")
    controller = (root / "judge/evaluate_hidden.py").read_text(encoding="utf-8")
    for fragment in ("judge_shards", "plan_shards("):
        if fragment not in controller:
            errors.append(f"judge/evaluate_hidden.py: missing {fragment!r}")
    judge = (root / "images/judge/Dockerfile").read_text(encoding="utf-8")
    copied = {part for line in judge.splitlines() if line.startswith("COPY ") and "/opt/qvg-judge/" in line
              for part in line.split()[1:-1] if not part.startswith("--")}
    for source in ("judge/evaluate_hidden.py", "judge/run_judge.py", "judge/judge_shards.py"):
        if source not in copied:
            errors.append(f"images/judge/Dockerfile must copy {source} into /opt/qvg-judge/")


def main() -> int:
    """Run every layout check and report."""
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parents[1]).resolve()
    errors: list[str] = []
    missing = [name for name in REQUIRED if name != RELEASE and not (root / name).is_file()]
    if not release_path(root).is_file():
        missing.append(f"{RELEASE} (or {release_path(root)}: run assets/fetch.sh)")
    if missing:
        errors.append("missing required files: " + ", ".join(missing))
    task = json.loads((root / f"tasks/{TASK_ID}.json").read_text(encoding="utf-8"))
    if task["task_id"] != TASK_ID:
        errors.append("task metadata ID differs")
    if task["submit_paths"] != ["submission"] or task["cwd"] != "/workspace":
        errors.append("submit contract differs")
    if not errors:
        validate_release(root, errors)
        validate_quality(root, errors)
        validate_private_inputs(root, errors)
        validate_self_contained_base(root, errors)
        validate_sharded_judge(root, errors)
    if errors:
        raise SystemExit("layout validation failed:\n  - " + "\n  - ".join(errors))
    release = json.loads(release_path(root).read_text(encoding="utf-8"))
    status = release["status"]
    if status != "calibrated":
        # The controller requires every golden and init digest in any status.
        unpinned = [f"{case['case_id']}.{key}" for case in release["cases"]
                    for key in ("bf16_frames_sha256", "init_video_sha256") if key in case and case[key] is None]
        if unpinned:
            shown = ", ".join(unpinned[:4]) + (", ..." if len(unpinned) > 4 else "")
            print(f"note: the release is {status} with {len(unpinned)} unpinned golden/init digests ({shown}); "
                  "the Judge refuses to run until they are re-bound from the staging MANIFEST.json, "
                  "the manifests refreshed and the Judge image rebuilt")
        else:
            print(f"note: the release is {status}: the Judge records every reading but returns valid=false")
    print(f"layout OK: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
