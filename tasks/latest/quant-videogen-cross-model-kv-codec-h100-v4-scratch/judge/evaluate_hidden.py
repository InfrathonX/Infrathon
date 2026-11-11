#!/usr/bin/env python3
"""Root-only controller for the scratch cross-model evaluation (v4).

The hidden suite is a case table (6 LongCat continuations of 10 segments and
11 HY-WorldPlay rollouts) frozen in a schema-2 release.  Cases are independent,
so the controller shards them across every GPU the Judge sees: one thread, one
candidate account (release ``judge.slot_uids[slot]``), one CPU slice and one device per
slot, cases run one after another inside a slot, heaviest first
(``judge_shards.plan_shards``).  Every GPU stays visible and a case is bound to
its slot's device by index (``QVG_DEVICE_INDEX`` in PCI bus order), never by
``CUDA_VISIBLE_DEVICES``.  The per-case computation does not depend on the
slot or on the GPU count, so neither does the score; only the wall time does.

Participant Python still runs behind a fixed per-slot UID/capability boundary
before root-owned metrics inspect its frozen outputs on the same device.  The
first candidate or infrastructure failure cancels every slot.  The score is
quality-weighted compression: the geometric mean over models of
``R * (1 + q)``, where ``R = sum(bf16_kv_bytes) / sum(stored_kv_bytes)`` across
that model's cases and ``q`` rewards (or penalizes) generation quality
relative to the official Quant-VideoGen codec (``quality_terms``).  It is
gated by per-model mean fidelity/VBench gates and looser per-case floors; the
latency gate of a case is the stage budget the controller enforced on it.
The official codec's readings are the zero point of ``q``, never an
admission threshold.  They are measured on each supported device class and
frozen per class in a quality file beside the release (``--quality``; the
asset ``hw/<type>/judge/h100_scratch_v4_quality.json``), so the release, its
gates and its BF16 readings stay the same on every class.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pwd
import re
import shutil
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import judge_shards


TASK_ID = "quant_videogen_cross_model_kv_codec_h100_scratch_v4"
# 5: ``score`` is the quality-weighted compression score; the plain
# compression geometric mean moved to ``compression_score``.
RESULT_SCHEMA_VERSION = 5
# Root-private per-case control records, rewritten whenever a case changes
# state, so a controller the adapter had to kill still leaves them behind.
PROGRESS_NAME = "qvg-progress.json"
PROTOCOL_SCHEMA_VERSION = 2
RELEASE_STATUSES = ("calibrated", "calibrating")
# Names the Judge image gives the slot UIDs (slot 0 keeps the v3 name).  They
# are provenance only: accounts are checked by UID/GID, never by name.
SLOT_ACCOUNTS = ("qvg-candidate", *(f"qvg-candidate-{slot}" for slot in range(2, 9)))
DEFAULT_SLOT_UIDS = tuple(range(23001, 23009))
MAX_OUTPUT_ENTRIES = 4096
# Per case.  Eight cases run at once, so the cap bounds Judge scratch at
# 8 x 4 GiB; a trusted 10-segment LongCat output is about 0.34 GB.
MAX_OUTPUT_BYTES = 4 * 1024 * 1024 * 1024
SOURCE_POLICY = Path("/opt/qvg-policy/codec_source_policy.py")
LONGCAT_RUNNER = Path("/opt/qvg-runners/run_longcat.py")
HY_RUNNER = Path("/opt/qvg-runners/run_hy_worldplay_8b.py")
PAPER_PROTOCOL = Path("/opt/mlsysforge/configs/paper_protocol.json")
COMPARE_FRAMES = Path("/opt/mlsysforge/metrics/compare_frames.py")
RUN_VBENCH = Path("/opt/mlsysforge/metrics/run_vbench.py")
HIDDEN_ROOT = Path("/eval/hidden")
CANDIDATE_CWD = Path("/workspace")
CGROUP_ROOT = Path("/sys/fs/cgroup")
PRIVATE_PATHS = (
    Path("/eval"),
    Path("/opt/qvg-judge"),
    Path("/opt/qvg-policy"),
    Path("/opt/mlsysforge/configs"),
)
# The slot preflight proves each candidate account can read what the runners
# need and cannot read what only the controller may.
SLOT_READABLE_PATHS = (
    Path("/models/hidden/longcat"),
    Path("/models/hidden/hy-worldplay"),
    Path("/opt/qvg-runners"),
    Path("/opt/qvg-src"),
    Path("/workspace/submission"),
)
SLOT_PRIVATE_PATHS = (*PRIVATE_PATHS, Path("/opt/mlsysforge/metrics"))

MODELS = ("longcat", "hy_worldplay_8b")
# Where each trusted runner leaves profile.json and the frame archive inside
# the candidate output (LongCat names its run directory <prompt_idx>-<seed>
# and the controller always stages a one-line prompt file).
RUN_DIRS = {"longcat": "1-{seed}", "hy_worldplay_8b": ""}
FRAME_HEIGHT = 480
FRAME_WIDTH = 832
VBENCH_DIMENSIONS = (
    "aesthetic_quality",
    "background_consistency",
    "image_quality",
    "subject_consistency",
)
FIDELITY_METRICS = ("psnr", "ssim", "lpips")
# PSNR has no finite ideal (identical frames are +inf, 100 dB in
# compare_frames.py), so its gain counts up to this ceiling.
PSNR_CEILING_DB = 40.0
# q = FIDELITY_WEIGHT * f + VIDEO_WEIGHT * v.
FIDELITY_WEIGHT = 0.5
VIDEO_WEIGHT = 0.5
HY_IMAGE_SUFFIXES = (".png", ".jpeg", ".jpg")
CASE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
POSE = re.compile(r"[a-z]+-[0-9]+(,[a-z]+-[0-9]+)*")
SHA256 = re.compile(r"[0-9a-f]{64}")

# Capability envelope, not a device model.  BF16 storage and the plain
# CUDA-tensor codec ABI need nothing past sm_80.  The memory floor is a
# release field (``judge.min_device_memory_bytes``): a BF16 HY rollout peaks
# at 90.7 GiB (a 10-segment LongCat BF16 continuation has not yet been
# measured to completion), so only the 94 GB-and-larger class can host every
# case.  The GPU model, UUID and total memory are recorded as measurement
# provenance only and never gate anything.
MIN_COMPUTE_CAPABILITY = (8, 0)
PROBE_SECONDS = 120
SLOT_PREFLIGHT_SECONDS = 180
CLEANUP_SECONDS = 10.0
POLL_SECONDS = 0.5


class CandidateFailure(RuntimeError):
    """Participant-controlled source or execution failed its contract."""


class InfrastructureFailure(RuntimeError):
    """Trusted runtime, mount, release, or scorer is invalid."""


def _strict_json(path: Path, *, candidate: bool = False) -> dict[str, Any]:
    failure = CandidateFailure if candidate else InfrastructureFailure
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("missing, unsafe, or oversized JSON")
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise failure(f"JSON contract failed for {path.name}") from exc
    if not isinstance(value, dict):
        raise failure(f"JSON root is not an object: {path.name}")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise InfrastructureFailure("a frozen artifact cannot be read") from exc
    return digest.hexdigest()


def _finite(value: Any, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateFailure(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise CandidateFailure(f"{context} is outside its valid range")
    return result


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CandidateFailure(f"{context} must be a positive integer")
    return value


# --------------------------------------------------------------------------
# Release (protocol schema 2)
# --------------------------------------------------------------------------


def _release_int(value: Any, context: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InfrastructureFailure(f"release field {context} must be an integer >= {minimum}")
    return value


def _release_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise InfrastructureFailure(f"release field {context} must be a finite number")
    return float(value)


def _release_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InfrastructureFailure(f"release field {context} must be a non-empty string")
    return value


def _release_sha(value: Any, context: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise InfrastructureFailure(f"release field {context} must be a sha256 hex digest")
    return value


def _release_path(value: Any, context: str, hidden_root: Path) -> Path:
    text = _release_text(value, context)
    path = Path(text)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(hidden_root):
        raise InfrastructureFailure(f"release path {context} must live under {hidden_root}")
    return path


def _release_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InfrastructureFailure(f"release field {context} must be an object")
    return value


def _dimensions(value: Any, context: str) -> dict[str, float]:
    mapping = _release_object(value, context)
    if set(mapping) != set(VBENCH_DIMENSIONS):
        raise InfrastructureFailure(f"release field {context} must name exactly the VBench dimensions")
    return {key: _release_number(mapping[key], f"{context}.{key}") for key in VBENCH_DIMENSIONS}


def _gate_set(value: Any, context: str) -> dict[str, float]:
    gates = _release_object(value, context)
    for key in ("psnr_min", "ssim_min", "lpips_max"):
        _release_number(gates.get(key), f"{context}.{key}")
    return _dimensions(gates.get("vbench_allowed_drop"), f"{context}.vbench_allowed_drop")


def _quality_reference(value: Any, context: str) -> dict[str, Any]:
    """Case means of the reference codec on one model: the zero point of ``q``."""
    reference = _release_object(value, context)
    if set(reference) != {*FIDELITY_METRICS, "vbench"}:
        raise InfrastructureFailure(f"release field {context} must name exactly psnr, ssim, lpips and vbench")
    psnr, ssim, lpips = (_release_number(reference[key], f"{context}.{key}") for key in FIDELITY_METRICS)
    if not (psnr < PSNR_CEILING_DB and 0.0 < ssim < 1.0 and lpips > 0.0):
        raise InfrastructureFailure(f"release field {context} must lie strictly inside the fidelity ideals")
    _dimensions(reference["vbench"], f"{context}.vbench")
    return reference


def validate_release(release: Any, *, hidden_root: Path = HIDDEN_ROOT) -> dict[str, Any]:
    """Validate the frozen schema-2 case table without touching the filesystem."""

    release = _release_object(release, "release")
    if release.get("status") not in RELEASE_STATUSES:
        raise InfrastructureFailure("the frozen release is neither calibrated nor calibrating")
    if release.get("protocol_schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise InfrastructureFailure("the frozen release is not protocol schema 2")
    calibrated = release["status"] == "calibrated"
    _release_sha(release.get("paper_protocol_sha256"), "paper_protocol_sha256")
    metrics = _release_object(release.get("metric_implementation_sha256"), "metric_implementation_sha256")
    for key in ("fidelity", "vbench"):
        _release_sha(metrics.get(key), f"metric_implementation_sha256.{key}")
    _release_path(release.get("prompt_file"), "prompt_file", hidden_root)
    _release_sha(release.get("prompt_file_sha256"), "prompt_file_sha256")

    judge = _release_object(release.get("judge"), "judge")
    for key in (
        "max_gpu_count",
        "min_gpu_count",
        "master_port_base",
        "threads_per_slot",
        "case_overhead_seconds",
        "controller_overhead_seconds",
        "min_device_memory_bytes",
    ):
        _release_int(judge.get(key), f"judge.{key}")
    _release_int(judge.get("start_stagger_seconds"), "judge.start_stagger_seconds", minimum=0)
    if not 1 <= judge["min_gpu_count"] <= judge["max_gpu_count"] <= judge_shards.MAX_SHARDS:
        raise InfrastructureFailure("the release GPU range is invalid")
    uids = judge.get("slot_uids")
    if (
        not isinstance(uids, list)
        or len(uids) < judge["max_gpu_count"]
        or any(isinstance(uid, bool) or not isinstance(uid, int) or not 1000 <= uid < 65534 for uid in uids)
        or len(set(uids)) != len(uids)
    ):
        raise InfrastructureFailure("the release must give every slot its own unprivileged UID")

    models = _release_object(release.get("models"), "models")
    if set(models) != set(MODELS):
        raise InfrastructureFailure("the release must define exactly the scored models")
    for name in MODELS:
        model = _release_object(models[name], f"models.{name}")
        for key in ("fps", "stage_seconds", "fidelity_metric_seconds", "vbench_metric_seconds"):
            _release_int(model.get(key), f"models.{name}.{key}")
        if not re.fullmatch(r"[a-z_]+", str(model.get("profile_key", ""))):
            raise InfrastructureFailure(f"release field models.{name}.profile_key is invalid")
        frames_file = model.get("frames_file")
        if not isinstance(frames_file, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.pt", frames_file):
            raise InfrastructureFailure(f"release field models.{name}.frames_file is invalid")
        gates = _release_object(model.get("gates"), f"models.{name}.gates")
        for kind in ("mean", "case_floor"):
            drops = _gate_set(gates.get(kind), f"models.{name}.gates.{kind}")
            # The quality term divides by the mean gate's drops.
            if kind == "mean" and min(drops.values()) <= 0:
                raise InfrastructureFailure(f"release field models.{name}.gates.mean.vbench_allowed_drop must be positive")
        # The zero point of q is per device class (apply_quality); evaluate()
        # refuses a calibrated release that reaches it without one.
        if model.get("quality_reference") is not None:
            _quality_reference(model["quality_reference"], f"models.{name}.quality_reference")

    cases = release.get("cases")
    if not isinstance(cases, list) or not cases:
        raise InfrastructureFailure("the release declares no cases")
    if judge["master_port_base"] < 1024 or judge["master_port_base"] + len(cases) - 1 > 65535:
        raise InfrastructureFailure("the release master port range is invalid")
    seen: set[str] = set()
    for position, case in enumerate(cases):
        case = _release_object(case, f"cases[{position}]")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not CASE_ID.fullmatch(case_id) or case_id in seen:
            raise InfrastructureFailure(f"release case {position} has a missing, unsafe, or duplicate case_id")
        seen.add(case_id)
        context = f"cases.{case_id}"
        name = case.get("model")
        if name not in MODELS:
            raise InfrastructureFailure(f"release field {context}.model is unknown")
        _release_int(case.get("seed"), f"{context}.seed", minimum=0)
        prompt = _release_text(case.get("prompt"), f"{context}.prompt")
        if "\n" in prompt or "\r" in prompt:
            raise InfrastructureFailure(f"release field {context}.prompt must be one line")
        frame_count = _release_int(case.get("frame_count"), f"{context}.frame_count", minimum=2)
        window = case.get("fidelity_frames")
        if (
            not isinstance(window, list)
            or len(window) != 2
            or any(isinstance(x, bool) or not isinstance(x, int) for x in window)
            or not 1 <= window[0] < window[1] <= frame_count
        ):
            raise InfrastructureFailure(f"release field {context}.fidelity_frames is invalid")
        _release_path(case.get("bf16_frames"), f"{context}.bf16_frames", hidden_root)
        _release_sha(case.get("bf16_frames_sha256"), f"{context}.bf16_frames_sha256")
        _dimensions(case.get("bf16_vbench"), f"{context}.bf16_vbench")
        frozen_bytes = case.get("bf16_kv_bytes")
        if frozen_bytes is None:
            if calibrated:
                raise InfrastructureFailure(f"calibrated release case {case_id} has no frozen bf16_kv_bytes")
        else:
            _release_int(frozen_bytes, f"{context}.bf16_kv_bytes")
        if name == "longcat":
            _release_int(case.get("prompt_index"), f"{context}.prompt_index")
            _release_int(case.get("segments"), f"{context}.segments")
            _release_path(case.get("init_video"), f"{context}.init_video", hidden_root)
            _release_sha(case.get("init_video_sha256"), f"{context}.init_video_sha256")
        else:
            image = _release_path(case.get("image"), f"{context}.image", hidden_root)
            if image.suffix.lower() not in HY_IMAGE_SUFFIXES:
                raise InfrastructureFailure(f"release field {context}.image is not an image")
            _release_sha(case.get("image_sha256"), f"{context}.image_sha256")
            length = _release_int(case.get("video_length"), f"{context}.video_length", minimum=5)
            if length != frame_count or (length - 1) % 4:
                raise InfrastructureFailure(f"release field {context}.video_length is invalid")
            if not POSE.fullmatch(_release_text(case.get("pose"), f"{context}.pose")):
                raise InfrastructureFailure(f"release field {context}.pose is invalid")
    for name in MODELS:
        if not any(case["model"] == name for case in cases):
            raise InfrastructureFailure(f"the release has no {name} case")
    return release


QUALITY_KEYS = {"schema_version", "hardware", "models", "oracle_calibration"}


def apply_quality(release: Any, quality: Any) -> dict[str, Any]:
    """Insert one device class's quality file into the release document.

    The file carries, per model, the reference codec's case means measured on
    that class (``models.<name>.quality_reference``) plus its oracle provenance.
    The release must not carry a reference of its own, so exactly one source
    sets the zero point of q.
    """
    release = _release_object(release, "release")
    quality = _release_object(quality, "quality")
    if not set(quality) <= QUALITY_KEYS or not {"schema_version", "hardware", "models"} <= set(quality):
        raise InfrastructureFailure(f"the quality file must hold {sorted(QUALITY_KEYS)}")
    if quality["schema_version"] != 1 or not isinstance(quality["hardware"], str) or not quality["hardware"]:
        raise InfrastructureFailure("the quality file must be schema 1 and name its hardware type")
    models = _release_object(release.get("models"), "models")
    entries = _release_object(quality["models"], "quality.models")
    if set(entries) != set(models):
        raise InfrastructureFailure(f"the quality file must name exactly the models {sorted(models)}")
    for name, entry in entries.items():
        entry = _release_object(entry, f"quality.models.{name}")
        if set(entry) != {"quality_reference"}:
            raise InfrastructureFailure(f"quality.models.{name} must hold exactly quality_reference")
        model = _release_object(models[name], f"models.{name}")
        if "quality_reference" in model:
            raise InfrastructureFailure(f"the release itself carries models.{name}.quality_reference")
        model["quality_reference"] = _quality_reference(entry["quality_reference"],
                                                        f"quality.models.{name}.quality_reference")
    return release


def require_quality(release: dict[str, Any]) -> None:
    """A calibrated release scores only with its device class's quality reference."""
    missing = [name for name, model in release["models"].items() if model.get("quality_reference") is None]
    if release["status"] == "calibrated" and missing:
        raise InfrastructureFailure(
            f"calibrated release without a quality file for this device class (no quality_reference for {missing})")


def load_release(path: Path, *, quality_path: Path | None = None,
                 hidden_root: Path = HIDDEN_ROOT) -> dict[str, Any]:
    release = _strict_json(path)
    if quality_path is not None:
        release = apply_quality(release, _strict_json(quality_path))
    return validate_release(release, hidden_root=hidden_root)


# --------------------------------------------------------------------------
# Execution budget
# --------------------------------------------------------------------------


def model_unit_seconds(release: dict[str, Any], name: str) -> int:
    """Upper bound of one case of ``name`` on its slot.

    The candidate stage is killed at ``stage_seconds``; fidelity and VBench run
    concurrently, each killed at its own budget; ``case_overhead_seconds``
    covers staging, the per-slot cleanups, freezing and the profile checks.
    """
    model = release["models"][name]
    return (
        model["stage_seconds"]
        + max(model["fidelity_metric_seconds"], model["vbench_metric_seconds"])
        + release["judge"]["case_overhead_seconds"]
    )


def _case_costs(release: dict[str, Any]) -> dict[str, int]:
    return {case["case_id"]: model_unit_seconds(release, case["model"]) for case in release["cases"]}


def _stagger_offsets(release: dict[str, Any], gpu_count: int) -> list[int]:
    return [slot * release["judge"]["start_stagger_seconds"] for slot in range(gpu_count)]


def schedule_plan(release: dict[str, Any], gpu_count: int) -> list[list[str]]:
    """Static heaviest-first assignment of cases to slots (release order inside a slot)."""
    if not 1 <= gpu_count <= judge_shards.MAX_SHARDS:
        raise InfrastructureFailure("the Judge GPU count is outside the shard range")
    return judge_shards.plan_shards(
        [case["case_id"] for case in release["cases"]],
        _case_costs(release),
        gpu_count,
        initial=_stagger_offsets(release, gpu_count),
    )


def controller_budget_seconds(release: dict[str, Any], gpu_count: int) -> int:
    """Derive the controller bound from the budgets the controller enforces.

    A list-scheduling simulation of the static plan on ``gpu_count`` slots in
    which every case takes its full unit budget and slot ``i`` starts after
    ``i * start_stagger_seconds``, plus ``controller_overhead_seconds`` for the
    release hashes, the hardware probe, admission and the slot preflight.
    Because each slot runs a fixed list, no case finishing early can make the
    run finish later, so the bound holds for every admitted run.
    """
    costs = _case_costs(release)
    initial = _stagger_offsets(release, gpu_count)
    makespan = judge_shards.plan_makespan(schedule_plan(release, gpu_count), costs, initial)
    return int(math.ceil(makespan)) + release["judge"]["controller_overhead_seconds"]


def execution_budget(release: dict[str, Any]) -> dict[str, Any]:
    """Summarize the wall budgets the controller enforces (private provenance)."""
    judge = release["judge"]
    return {
        "models": {
            name: {
                "stage_seconds": release["models"][name]["stage_seconds"],
                "fidelity_metric_seconds": release["models"][name]["fidelity_metric_seconds"],
                "vbench_metric_seconds": release["models"][name]["vbench_metric_seconds"],
                "case_unit_seconds": model_unit_seconds(release, name),
            }
            for name in MODELS
        },
        "case_overhead_seconds": judge["case_overhead_seconds"],
        "start_stagger_seconds": judge["start_stagger_seconds"],
        "controller_overhead_seconds": judge["controller_overhead_seconds"],
        "controller_seconds_by_gpu_count": {
            str(count): controller_budget_seconds(release, count)
            for count in range(judge["min_gpu_count"], judge["max_gpu_count"] + 1)
        },
    }


# --------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------

DEVICE_PROBE = r"""
import json
import torch
rows = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        free, _total = torch.cuda.mem_get_info(index)
        pci = [getattr(props, key, None) for key in ("pci_domain_id", "pci_bus_id", "pci_device_id")]
        rows.append({
            "index": index,
            "name": str(props.name),
            "uuid": str(getattr(props, "uuid", "")),
            "capability": [int(props.major), int(props.minor)],
            "total_memory_bytes": int(props.total_memory),
            "free_memory_bytes": int(free),
            "pci_bus_id": "" if None in pci else "%08x:%02x:%02x.0" % tuple(pci),
        })
print(json.dumps({"devices": rows, "torch": torch.__version__}))
"""


def _device_environment() -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if key != "CUDA_VISIBLE_DEVICES"}
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return environment


def _last_json_line(text: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("no output")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise ValueError("output is not an object")
    return value


def probe_devices(*, timeout: int = PROBE_SECONDS) -> list[dict[str, Any]]:
    """Inventory every visible GPU in a child process.

    The probe sees the identity device mapping in PCI bus order, exactly what
    candidates and metrics see, and its CUDA contexts end with it, so the
    controller itself never holds GPU memory.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", DEVICE_PROBE],
            env=_device_environment(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"device probe exited {completed.returncode}")
        rows = _last_json_line(completed.stdout)["devices"]
        if not isinstance(rows, list):
            raise ValueError("device rows are not a list")
        return rows
    except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError, KeyError) as exc:
        raise InfrastructureFailure("the Judge GPU inventory could not be measured") from exc


def judge_fleet(
    rows: Sequence[Mapping[str, Any]],
    release: dict[str, Any],
    environ: Mapping[str, str] = os.environ,
) -> tuple[list[judge_shards.Gpu], dict[str, Any]]:
    """Admit min..max identical, idle GPUs that meet the capability envelope."""
    judge = release["judge"]
    try:
        gpus = sorted((judge_shards.gpu_from_probe(row) for row in rows), key=lambda gpu: gpu.index)
        gpus = judge_shards._visible_filter(gpus, environ)
        fleet = judge_shards.validate_fleet(
            gpus,
            min_count=judge["min_gpu_count"],
            max_count=judge["max_gpu_count"],
            min_capability=MIN_COMPUTE_CAPABILITY,
            min_memory_bytes=judge["min_device_memory_bytes"],
        )
    except RuntimeError as exc:
        raise InfrastructureFailure(f"the Judge GPUs are outside this task's envelope: {exc}") from exc
    fleet.update(
        min_compute_capability=list(MIN_COMPUTE_CAPABILITY),
        min_device_memory_bytes=judge["min_device_memory_bytes"],
    )
    return gpus, fleet


def evaluator_hardware(release: dict[str, Any]) -> tuple[list[judge_shards.Gpu], dict[str, Any]]:
    """Admit the fleet before staging anything, so an undersized or busy card
    fails fast as an infrastructure error instead of mid-run as an opaque
    candidate OOM."""
    return judge_fleet(probe_devices(), release)


# --------------------------------------------------------------------------
# Isolation and process control
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tools:
    setpriv: str
    pkill: str
    taskset: str


@dataclass(frozen=True)
class Slot:
    index: int
    gpu: judge_shards.Gpu
    uid: int
    cpus: tuple[int, ...]

    def record(self) -> dict[str, Any]:
        return {
            "slot": self.index,
            "uid": self.uid,
            "account": slot_account_name(self.index),
            "gpu_index": self.gpu.index,
            "gpu_uuid": self.gpu.uuid,
            "cpus": list(self.cpus),
        }


def slot_account_name(slot: int) -> str:
    return SLOT_ACCOUNTS[slot] if slot < len(SLOT_ACCOUNTS) else f"uid-{slot}"


def isolation_prerequisites(release: dict[str, Any]) -> Tools:
    if os.geteuid() != 0:
        raise InfrastructureFailure("the protected controller must run as root")
    for uid in release["judge"]["slot_uids"]:
        try:
            account = pwd.getpwuid(uid)
        except KeyError as exc:
            raise InfrastructureFailure("an isolated candidate account is missing") from exc
        if account.pw_uid != uid or account.pw_gid != uid:
            raise InfrastructureFailure("an isolated candidate account has unexpected IDs")
    tools = {name: shutil.which(name) for name in ("setpriv", "pkill", "taskset")}
    if any(path is None for path in tools.values()):
        raise InfrastructureFailure("candidate isolation tools are missing")
    for private in PRIVATE_PATHS:
        try:
            info = private.stat()
        except OSError as exc:
            raise InfrastructureFailure("a private evaluator path is missing") from exc
        if info.st_uid != 0 or info.st_mode & 0o007 or (info.st_gid != 0 and info.st_mode & 0o070):
            raise InfrastructureFailure("a private evaluator path is candidate-readable")
    return Tools(**tools)  # type: ignore[arg-type]


def candidate_command(setpriv: str, uid: int, command: list[str]) -> list[str]:
    return [
        setpriv,
        f"--reuid={uid}",
        f"--regid={uid}",
        "--clear-groups",
        "--no-new-privs",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--bounding-set=-all",
        "--pdeathsig=KILL",
        "--",
        *command,
    ]


def candidate_environment(stage: Path, slot: Slot) -> dict[str, str]:
    allowed = {
        "CUDA_HOME",
        "LD_LIBRARY_PATH",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "PATH",
        # Deployment switches for the trusted runners (PyroMind: one up-front
        # CUDA reservation per slot; see run_longcat.build_env).
        "QVG_CUDA_RESERVE",
        "QVG_CUDA_RESERVE_HEADROOM_BYTES",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    environment.update(
        {
            "HOME": str(stage / "home"),
            "TMPDIR": str(stage / "tmp"),
            "HF_HOME": str(stage / "cache" / "huggingface"),
            "TORCH_HOME": "/opt/vbench-cache/torch",
            "TORCH_EXTENSIONS_DIR": str(stage / "cache" / "torch-extensions"),
            "TRITON_CACHE_DIR": str(stage / "cache" / "triton"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    # QVG_DEVICE_INDEX / CUDA_DEVICE_ORDER / thread counts; never
    # CUDA_VISIBLE_DEVICES (see judge_shards.shard_environment).
    return judge_shards.shard_environment(environment, slot.gpu, slot.cpus)


def metric_environment(slot: Slot) -> dict[str, str]:
    return judge_shards.shard_environment(os.environ, slot.gpu, slot.cpus)


def _live_uid_pids(uids: Sequence[int]) -> list[int]:
    wanted = set(uids)
    result: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "status").read_text(encoding="utf-8").splitlines()
            uid_line = next(line for line in fields if line.startswith("Uid:"))
            state_line = next(line for line in fields if line.startswith("State:"))
            uid = int(uid_line.split()[1])
        except (OSError, StopIteration, ValueError, IndexError):
            continue
        if uid in wanted and "Z (zombie)" not in state_line:
            result.append(int(entry.name))
    return result


class ProcessControl:
    """Kill candidate processes by slot UID.

    A slot cleans only its own UID before and after each of its cases; every
    slot UID is killed only at the start and end of the run and on cancel.
    """

    def __init__(self, pkill: str) -> None:
        self.pkill = pkill

    def kill_uids(self, uids: Sequence[int]) -> None:
        if not uids:
            return
        try:
            subprocess.run(
                [self.pkill, "-KILL", "-u", ",".join(str(uid) for uid in uids)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InfrastructureFailure("candidate processes could not be cleared") from exc
        deadline = time.monotonic() + CLEANUP_SECONDS
        while _live_uid_pids(uids):
            if time.monotonic() >= deadline:
                raise InfrastructureFailure("candidate processes did not stop")
            time.sleep(0.05)


class ChildRegistry:
    """Every child the controller starts, so a cancel can kill all of them.

    Candidates lead their own session; trusted metrics lead their own process
    group inside the controller's session (so ``run_judge`` can still reach
    them through that session if the controller itself is killed).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: set[int] = set()
        self._groups: set[int] = set()

    def add_session(self, pid: int) -> None:
        with self._lock:
            self._sessions.add(pid)

    def add_session_unless(self, pid: int, cancel: threading.Event | None) -> bool:
        """Register a session unless the run is cancelled; False when it is.

        A cancel sets its event before ``kill_all`` snapshots the sessions
        under this lock, so a session started around a cancel is either seen
        by that snapshot or refused here; it can never outlive the abort.
        """
        with self._lock:
            if cancel is not None and cancel.is_set():
                return False
            self._sessions.add(pid)
            return True

    def add_group(self, pid: int) -> None:
        with self._lock:
            self._groups.add(pid)

    def discard(self, pid: int) -> None:
        with self._lock:
            self._sessions.discard(pid)
            self._groups.discard(pid)

    def kill_all(self) -> None:
        with self._lock:
            sessions, groups = set(self._sessions), set(self._groups)
        for group in groups:
            try:
                os.killpg(group, signal.SIGKILL)
            except OSError:
                pass
        if not sessions:
            return
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                try:
                    if os.getsid(int(entry.name)) in sessions:
                        os.killpg(os.getpgid(int(entry.name)), signal.SIGKILL)
                except OSError:
                    pass


def cgroup_oom_kills(cgroup: Path = CGROUP_ROOT) -> int | None:
    """OOM kills charged to the Judge container's memory cgroup (v2, else v1)."""
    for path in (cgroup / "memory.events", cgroup / "memory" / "memory.oom_control"):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition(" ")
                if key == "oom_kill":
                    return int(value)
        except (OSError, ValueError):
            continue
    return None


def _kill_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        pass


# --------------------------------------------------------------------------
# Admission and frozen trees
# --------------------------------------------------------------------------


def _load_source_policy() -> Any:
    try:
        spec = importlib.util.spec_from_file_location("qvg_codec_source_policy", SOURCE_POLICY)
        if spec is None or spec.loader is None:
            raise ValueError("policy loader is unavailable")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        raise InfrastructureFailure("the canonical source policy cannot be loaded") from exc


def _walk_regular_tree(root: Path, *, candidate_output: bool) -> tuple[list[Path], list[Path], int]:
    failure = CandidateFailure
    if root.is_symlink() or not root.is_dir():
        raise failure("candidate tree is missing or unsafe")
    directories = [root]
    files: list[Path] = []
    total_bytes = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise failure("candidate tree cannot be inspected") from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise failure("candidate tree entry cannot be inspected") from exc
            if stat.S_ISLNK(info.st_mode):
                raise failure("links are forbidden in candidate trees")
            if stat.S_ISDIR(info.st_mode):
                directories.append(path)
                stack.append(path)
            elif stat.S_ISREG(info.st_mode):
                files.append(path)
                total_bytes += info.st_size
            else:
                raise failure("special files are forbidden in candidate trees")
            if len(directories) + len(files) > MAX_OUTPUT_ENTRIES:
                raise failure("candidate output has too many entries")
            if candidate_output and total_bytes > MAX_OUTPUT_BYTES:
                raise failure("candidate output exceeds its size limit")
    return directories, files, total_bytes


def _freeze_tree(root: Path, *, files_mode: int, dirs_mode: int, output: bool) -> None:
    directories, files, _ = _walk_regular_tree(root, candidate_output=output)
    for path in files:
        os.chown(path, 0, 0, follow_symlinks=False)
        os.chmod(path, files_mode, follow_symlinks=False)
    for path in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        os.chown(path, 0, 0, follow_symlinks=False)
        os.chmod(path, dirs_mode, follow_symlinks=False)


def admit_submission(submission: Path) -> dict[str, Any]:
    policy = _load_source_policy()
    try:
        result = policy.validate_submission(submission)
    except policy.PolicyError as exc:
        raise CandidateFailure("source admission rejected the submission") from exc
    except Exception as exc:
        raise InfrastructureFailure("source admission failed unexpectedly") from exc
    if not isinstance(result, dict) or result.get("valid") is not True:
        raise InfrastructureFailure("source admission returned an invalid result")
    _freeze_tree(submission, files_mode=0o444, dirs_mode=0o555, output=False)
    return result


def frozen_files(release: dict[str, Any]) -> dict[Path, str]:
    """Every trusted input the verdict depends on, with its pinned digest."""
    files: dict[Path, str] = {
        PAPER_PROTOCOL: release["paper_protocol_sha256"],
        COMPARE_FRAMES: release["metric_implementation_sha256"]["fidelity"],
        RUN_VBENCH: release["metric_implementation_sha256"]["vbench"],
    }

    def pin(path: str, digest: str) -> None:
        if files.setdefault(Path(path), digest) != digest:
            raise InfrastructureFailure("one release path is pinned to two digests")

    pin(release["prompt_file"], release["prompt_file_sha256"])
    for case in release["cases"]:
        pin(case["bf16_frames"], case["bf16_frames_sha256"])
        if case["model"] == "longcat":
            pin(case["init_video"], case["init_video_sha256"])
        else:
            pin(case["image"], case["image_sha256"])
    return files


def verify_frozen_files(files: Mapping[Path, str]) -> None:
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise InfrastructureFailure("a frozen evaluation artifact is missing or unsafe")
    with ThreadPoolExecutor(max_workers=8) as pool:
        digests = dict(zip(files, pool.map(sha256, files)))
    if any(digests[path] != expected for path, expected in files.items()):
        raise InfrastructureFailure("a frozen evaluation artifact does not match its release")


def prompt_lines(release: dict[str, Any]) -> list[str]:
    """Read the frozen prompt file and check every LongCat case's prompt line."""
    try:
        lines = Path(release["prompt_file"]).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise InfrastructureFailure("the frozen prompt file cannot be read") from exc
    for case in release["cases"]:
        if case["model"] != "longcat":
            continue
        index = case["prompt_index"]
        if index > len(lines) or lines[index - 1] != case["prompt"]:
            raise InfrastructureFailure("a selected prompt disagrees with the release")
    return lines


# --------------------------------------------------------------------------
# One case
# --------------------------------------------------------------------------


def runner_command(
    release: dict[str, Any],
    case: dict[str, Any],
    *,
    stage: Path,
    master_port: int,
    device_index: int,
) -> list[str]:
    """The trusted runner invocation for one case (spec section 4.2)."""
    output = stage / "output"
    if case["model"] == "longcat":
        return [
            sys.executable,
            str(LONGCAT_RUNNER),
            "--mode",
            "candidate",
            "--prompt-file",
            str(stage / "input" / "prompt.txt"),
            "--prompt-index",
            "1",
            "--seed",
            str(case["seed"]),
            "--output",
            str(output),
            "--init-video",
            str(stage / "input" / "init.mp4"),
            "--segments",
            str(case["segments"]),
            "--master-port",
            str(master_port),
            "--device-index",
            str(device_index),
        ]
    return [
        sys.executable,
        str(HY_RUNNER),
        "--mode",
        "candidate",
        "--seed",
        str(case["seed"]),
        "--video-length",
        str(case["video_length"]),
        "--prompt",
        case["prompt"],
        "--image-path",
        str(staged_image(case, stage)),
        "--pose",
        case["pose"],
        "--output",
        str(output),
        "--master-port",
        str(master_port),
        "--device-index",
        str(device_index),
    ]


def staged_image(case: dict[str, Any], stage: Path) -> Path:
    return stage / "input" / f"image{Path(case['image']).suffix.lower()}"


def metric_commands(
    release: dict[str, Any],
    case: dict[str, Any],
    *,
    candidate_frames: Path,
    output: Path,
    device_index: int,
) -> tuple[list[str], list[str]]:
    """Root fidelity and VBench invocations on the slot's own device."""
    model = release["models"][case["model"]]
    device = f"cuda:{device_index}"
    fidelity = [
        sys.executable,
        str(COMPARE_FRAMES),
        "--reference",
        case["bf16_frames"],
        "--candidate",
        str(candidate_frames),
        "--frame-range",
        str(case["fidelity_frames"][0]),
        str(case["fidelity_frames"][1]),
        "--output",
        str(output / "fidelity.json"),
        "--device",
        device,
    ]
    vbench = [
        sys.executable,
        str(RUN_VBENCH),
        "--frames",
        str(candidate_frames),
        "--prompt",
        case["prompt"],
        "--fps",
        str(model["fps"]),
        "--output-dir",
        str(output / "vbench"),
        "--name",
        case["case_id"],
        "--device",
        device,
    ]
    return fidelity, vbench


def _frame_shape(path: Path, *, candidate_controlled: bool) -> tuple[int, ...]:
    """Validate the weights-only RGB tensor ABI before trusted metrics run."""

    failure = CandidateFailure if candidate_controlled else InfrastructureFailure
    try:
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise ValueError("frame archive is not a regular file")
        import torch

        frames = torch.load(path, map_location="cpu", weights_only=True)
        if type(frames) is not torch.Tensor or frames.layout != torch.strided:
            raise TypeError("frame archive does not contain one dense tensor")
        if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("frame archive is not RGB uint8 [T,H,W,C]")
        shape = tuple(int(value) for value in frames.shape)
        if shape[0] < 2 or shape[1] <= 0 or shape[2] <= 0:
            raise ValueError("frame archive dimensions are empty")
        return shape
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        origin = "candidate" if candidate_controlled else "frozen reference"
        raise failure(f"{origin} frame ABI is invalid") from exc


def check_frames(case: dict[str, Any], candidate: Path) -> None:
    expected = (case["frame_count"], FRAME_HEIGHT, FRAME_WIDTH, 3)
    if _frame_shape(Path(case["bf16_frames"]), candidate_controlled=False) != expected:
        raise InfrastructureFailure("the frozen reference frames disagree with the release")
    if candidate.is_symlink() or not candidate.is_file():
        raise CandidateFailure("candidate frame output is missing or unsafe")
    if _frame_shape(candidate, candidate_controlled=True) != expected:
        raise CandidateFailure("candidate frame shape differs from the frozen reference")


def expected_compression_calls(case: dict[str, Any]) -> int:
    """Compression calls the trusted pipeline makes for one case.

    LongCat compresses its cache once per continued segment
    (``quantize_kv_cache`` in ``generate_vc``).  HY rebuilds and compresses its
    vision context once per 4-latent chunk after the first; few-step mode
    disables guidance, so there is no second (negative-branch) pass.
    """
    if case["model"] == "longcat":
        return case["segments"]
    latents = (case["video_length"] - 1) // 4 + 1
    return max(latents // 4 - 1, 0)


def profile_row(
    profile_path: Path, key: str, case: dict[str, Any], *, device_index: int
) -> dict[str, Any]:
    """Validate one case's profile (written inside the candidate process)."""
    profile = _strict_json(profile_path, candidate=True)
    stats = profile.get(key)
    if not isinstance(stats, dict):
        raise CandidateFailure("candidate profile is incomplete")
    # Both counters accumulate over every compression call of the whole
    # generation (all LongCat segments, all HY chunks).
    stored = _positive_int(stats.get("stored_kv_bytes"), "stored_kv_bytes")
    bf16 = _positive_int(stats.get("bf16_kv_bytes"), "bf16_kv_bytes")
    if stored > bf16:
        raise CandidateFailure("persistent state is not compressed")
    ratio = bf16 / stored
    if "kv_compression_ratio" in stats:
        reported = _finite(stats["kv_compression_ratio"], "kv_compression_ratio", minimum=1.0)
        if not math.isclose(reported, ratio, rel_tol=1e-6, abs_tol=1e-6):
            raise CandidateFailure("candidate byte counters are inconsistent")
    # The BF16 byte count depends only on shapes; once the release freezes it,
    # a candidate cannot inflate the denominator.
    if case["bf16_kv_bytes"] is not None and bf16 != case["bf16_kv_bytes"]:
        raise CandidateFailure("candidate BF16 KV byte count differs from the frozen release")
    other = profile.get("other_device_allocated_bytes")
    if isinstance(other, bool) or not isinstance(other, int) or other < 0:
        raise CandidateFailure("candidate profile does not report other-device memory")
    if other:
        raise CandidateFailure("candidate allocated memory on a GPU outside its slot")
    device = profile.get("cuda_device_index")
    if isinstance(device, bool) or not isinstance(device, int) or device != device_index:
        raise CandidateFailure("candidate ran on a GPU other than its slot's")
    # Totals that miss a call (a profile from a partial or altered rollout)
    # would score fewer segments or chunks than the frames contain.
    calls = stats.get("calls")
    if isinstance(calls, bool) or not isinstance(calls, int) or calls != expected_compression_calls(case):
        raise CandidateFailure("candidate profile does not cover every compression call")
    row: dict[str, Any] = {
        "compression_ratio": ratio,
        "stored_kv_bytes": stored,
        "bf16_kv_bytes": bf16,
        "other_device_allocated_bytes": other,
        # Keep the frozen release's original measurement boundary.
        "wall_seconds": _finite(profile.get("wall_seconds"), "wall_seconds", minimum=0.0),
        "compression_calls": calls,
    }
    return row


def _fidelity_reading(value: dict[str, Any]) -> dict[str, Any]:
    reading: dict[str, Any] = {}
    for key in FIDELITY_METRICS:
        number = value.get(key)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
            raise InfrastructureFailure("a trusted fidelity metric is missing or non-finite")
        reading[key] = float(number)
    for key in ("evaluation_start_frame", "evaluation_end_frame", "first_divergent_frame"):
        if isinstance(value.get(key), int):
            reading[key] = value[key]
    return reading


def _vbench_reading(value: dict[str, Any]) -> dict[str, float]:
    dimensions = value.get("dimensions")
    if not isinstance(dimensions, dict):
        raise InfrastructureFailure("the trusted VBench result is incomplete")
    reading: dict[str, float] = {}
    for key in VBENCH_DIMENSIONS:
        number = dimensions.get(key)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
            raise InfrastructureFailure("a trusted VBench dimension is missing or non-finite")
        reading[key] = float(number)
    return reading


class RunState:
    """Fail-fast state shared by the slot threads."""

    def __init__(self, abort: Callable[[], None]) -> None:
        self._abort = abort
        self._lock = threading.Lock()
        self.cancel = threading.Event()
        self.first_error: dict[str, Any] | None = None
        self.started = time.monotonic()

    def now(self) -> float:
        return round(time.monotonic() - self.started, 3)

    def wait(self, seconds: float) -> bool:
        """Sleep up to ``seconds``; True when the run was cancelled meanwhile."""
        return self.cancel.wait(seconds) if seconds > 0 else self.cancel.is_set()

    def fail(self, kind: str, case_id: str, slot: int, message: str) -> bool:
        """Record the first failure and cancel every slot; False if one was recorded already."""
        with self._lock:
            if self.first_error is not None:
                return False
            self.first_error = {
                "kind": kind,
                "case_id": case_id,
                "slot": slot,
                "message": message,
                "at_seconds": self.now(),
            }
            self.cancel.set()
        self._abort()
        return True


def failure_kind(exc: BaseException) -> str:
    return "candidate_failure" if isinstance(exc, CandidateFailure) else "infrastructure_failure"


def _failure_message(exc: BaseException) -> str:
    if isinstance(exc, (CandidateFailure, InfrastructureFailure)):
        return str(exc)
    return f"unexpected protected evaluator failure: {type(exc).__name__}"


class CaseRunner:
    """Run one case on one slot: stage, candidate, cleanup, freeze, checks, metrics."""

    candidate_cwd = CANDIDATE_CWD
    cgroup_root = CGROUP_ROOT

    def __init__(
        self,
        release: dict[str, Any],
        *,
        tools: Tools,
        control: ProcessControl,
        registry: ChildRegistry,
        log_root: Path,
        scratch: Path,
        prompts: Sequence[str],
    ) -> None:
        self.release = release
        self.tools = tools
        self.control = control
        self.registry = registry
        self.log_root = log_root
        self.scratch = scratch
        self.prompts = list(prompts)
        self.oom_baseline = cgroup_oom_kills(self.cgroup_root)

    # -- orchestration -----------------------------------------------------

    def run_case(
        self, case: dict[str, Any], slot: Slot, control: dict[str, Any], state: RunState
    ) -> dict[str, Any]:
        model = self.release["models"][case["model"]]
        logs = self.case_log_dir(case)
        control["logs"] = {
            "candidate": str(logs / "candidate.log"),
            "fidelity": str(logs / "fidelity.log"),
            "vbench": str(logs / "vbench.log"),
        }
        stage = self.create_stage(slot)
        try:
            self.stage_inputs(case, stage, slot)
            command = runner_command(
                self.release,
                case,
                stage=stage,
                master_port=control["master_port"],
                device_index=slot.gpu.index,
            )
            # Only this slot's UID: the other slots' candidates keep running.
            self.control.kill_uids([slot.uid])
            control["stage_started_at"] = state.now()
            try:
                self.run_candidate(
                    command,
                    stage=stage,
                    slot=slot,
                    log=logs / "candidate.log",
                    timeout=model["stage_seconds"],
                    state=state,
                )
            finally:
                control["stage_ended_at"] = state.now()
                control["stage_wall_seconds"] = round(
                    control["stage_ended_at"] - control["stage_started_at"], 3
                )
                self.control.kill_uids([slot.uid])
            output = stage / "output"
            self.freeze_output(output)
            run_dir = output / RUN_DIRS[case["model"]].format(seed=case["seed"])
            row = profile_row(
                run_dir / "profile.json", model["profile_key"], case, device_index=slot.gpu.index
            )
            frames = run_dir / model["frames_file"]
            self.check_frames(case, frames)
            control["metrics_started_at"] = state.now()
            try:
                fidelity, vbench = self.run_metrics(case, slot, frames, logs, state)
            finally:
                control["metrics_ended_at"] = state.now()
            row.update(fidelity=fidelity, vbench=vbench)
            return row
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    # -- steps (overridable for CPU tests) ---------------------------------

    def case_log_dir(self, case: dict[str, Any]) -> Path:
        root = self.log_root / "cases"
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = root / case["case_id"]
        path.mkdir(mode=0o700, exist_ok=True)
        return path

    def create_stage(self, slot: Slot) -> Path:
        """A per-case stage only this slot's UID can traverse."""
        stage = Path(tempfile.mkdtemp(prefix=f"qvg-slot{slot.index}-", dir="/tmp"))
        os.chown(stage, 0, slot.uid)
        os.chmod(stage, 0o710)
        inputs = stage / "input"
        inputs.mkdir()
        os.chown(inputs, 0, slot.uid)
        os.chmod(inputs, 0o550)
        for name in ("output", "home", "tmp", "cache"):
            path = stage / name
            path.mkdir(exist_ok=False)
            os.chown(path, slot.uid, slot.uid)
            os.chmod(path, 0o700)
        return stage

    def _stage_file(self, source: Path, destination: Path, slot: Slot) -> None:
        if source.is_symlink() or not source.is_file():
            raise InfrastructureFailure("a selected workload input is unsafe")
        try:
            shutil.copyfile(source, destination)
            os.chown(destination, 0, slot.uid)
            os.chmod(destination, 0o440)
        except OSError as exc:
            raise InfrastructureFailure("a selected workload input cannot be staged") from exc

    def stage_inputs(self, case: dict[str, Any], stage: Path, slot: Slot) -> None:
        if case["model"] == "longcat":
            index = case["prompt_index"]
            if index > len(self.prompts) or self.prompts[index - 1] != case["prompt"]:
                raise InfrastructureFailure("the selected prompt disagrees with the release")
            selected = stage / "input" / "prompt.txt"
            try:
                selected.write_text(case["prompt"] + "\n", encoding="utf-8")
                os.chown(selected, 0, slot.uid)
                os.chmod(selected, 0o440)
            except OSError as exc:
                raise InfrastructureFailure("the selected prompt cannot be staged") from exc
            self._stage_file(Path(case["init_video"]), stage / "input" / "init.mp4", slot)
        else:
            self._stage_file(Path(case["image"]), staged_image(case, stage), slot)

    def _raise_if_cgroup_oom(self) -> None:
        if self.oom_baseline is None:
            return
        current = cgroup_oom_kills(self.cgroup_root)
        if current is not None and current > self.oom_baseline:
            raise InfrastructureFailure(
                "the Judge memory cgroup OOM-killed a process during the candidate stage"
            )

    def run_candidate(
        self,
        command: list[str],
        *,
        stage: Path,
        slot: Slot,
        log: Path,
        timeout: int,
        state: RunState | None = None,
    ) -> None:
        cancel = state.cancel if state is not None else None
        # Another slot's failure may have aborted the run while this slot was
        # staging; its one-shot abort never sees a candidate started later.
        if cancel is not None and cancel.is_set():
            raise InfrastructureFailure("the candidate stage was cancelled")
        full = judge_shards.taskset_command(
            self.tools.taskset, slot.cpus, candidate_command(self.tools.setpriv, slot.uid, command)
        )
        with log.open("w", encoding="utf-8") as handle:
            os.chmod(log, 0o600)
            try:
                process = subprocess.Popen(
                    full,
                    cwd=str(self.candidate_cwd),
                    env=candidate_environment(stage, slot),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            except OSError as exc:
                raise InfrastructureFailure("isolated candidate execution could not start") from exc
            deadline = time.monotonic() + timeout
            try:
                if not self.registry.add_session_unless(process.pid, cancel):
                    _kill_group(process)
                    process.wait(timeout=30)
                    raise InfrastructureFailure("the candidate stage was cancelled")
                # Short waits, so a run-wide cancel stops this candidate within
                # POLL_SECONDS instead of after its whole stage budget.
                while True:
                    try:
                        returncode = process.wait(
                            timeout=max(0.0, min(POLL_SECONDS, deadline - time.monotonic()))
                        )
                        break
                    except subprocess.TimeoutExpired as exc:
                        if cancel is not None and cancel.is_set():
                            _kill_group(process)
                            process.wait(timeout=30)
                            raise InfrastructureFailure("the candidate stage was cancelled") from exc
                        if time.monotonic() >= deadline:
                            _kill_group(process)
                            process.wait(timeout=30)
                            self._raise_if_cgroup_oom()
                            raise CandidateFailure("candidate execution exceeded its time limit") from exc
            finally:
                self.registry.discard(process.pid)
        if returncode:
            self._raise_if_cgroup_oom()
            try:
                diagnostic = log.read_text(encoding="utf-8", errors="replace")[-8192:]
            except OSError:
                diagnostic = "candidate log was unavailable"
            # This detail is written only to the root-private artifact. The
            # public adapter deliberately projects a generic candidate-error
            # message.
            raise CandidateFailure(
                f"candidate execution returned exit {returncode}; private log tail:\n{diagnostic}"
            )

    def freeze_output(self, output: Path) -> None:
        _freeze_tree(output, files_mode=0o400, dirs_mode=0o500, output=True)

    def check_frames(self, case: dict[str, Any], frames: Path) -> None:
        check_frames(case, frames)

    def run_metrics(
        self, case: dict[str, Any], slot: Slot, frames: Path, logs: Path, state: RunState
    ) -> tuple[dict[str, Any], dict[str, float]]:
        model = self.release["models"][case["model"]]
        output = self.scratch / case["case_id"]
        output.mkdir(mode=0o700)
        fidelity, vbench = metric_commands(
            self.release, case, candidate_frames=frames, output=output, device_index=slot.gpu.index
        )
        supervise_root(
            [
                (judge_shards.taskset_command(self.tools.taskset, slot.cpus, fidelity),
                 logs / "fidelity.log", model["fidelity_metric_seconds"]),
                (judge_shards.taskset_command(self.tools.taskset, slot.cpus, vbench),
                 logs / "vbench.log", model["vbench_metric_seconds"]),
            ],
            env=metric_environment(slot),
            registry=self.registry,
            state=state,
        )
        return (
            _fidelity_reading(_strict_json(output / "fidelity.json")),
            _vbench_reading(_strict_json(output / "vbench" / f"{case['case_id']}-paper-metrics.json")),
        )

    # -- slot preflight ----------------------------------------------------

    def preflight(self, slots: Sequence[Slot]) -> list[dict[str, Any]]:
        """Prove, before any case runs, that every slot account binds its own
        GPU, sees no CUDA_VISIBLE_DEVICES, can read the model mounts and
        cannot read the controller's private paths.  A platform that breaks
        any of this would otherwise surface as a candidate failure."""
        directory = self.log_root / "preflight"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            return list(pool.map(lambda slot: self._preflight_slot(slot, directory), slots))

    def _preflight_slot(self, slot: Slot, directory: Path) -> dict[str, Any]:
        stage = self.create_stage(slot)
        log = directory / f"slot-{slot.index}.log"
        expected = {
            "private": [str(path) for path in SLOT_PRIVATE_PATHS],
            "readable": [str(path) for path in SLOT_READABLE_PATHS],
        }
        command = judge_shards.taskset_command(
            self.tools.taskset,
            slot.cpus,
            candidate_command(
                self.tools.setpriv,
                slot.uid,
                [sys.executable, "-I", "-c", SLOT_PROBE, json.dumps(expected)],
            ),
        )
        try:
            self.control.kill_uids([slot.uid])
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(self.candidate_cwd),
                    env=candidate_environment(stage, slot),
                    capture_output=True,
                    text=True,
                    timeout=SLOT_PREFLIGHT_SECONDS,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise InfrastructureFailure(f"slot {slot.index} preflight could not complete") from exc
            finally:
                self.control.kill_uids([slot.uid])
            log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
            os.chmod(log, 0o600)
            try:
                seen = _last_json_line(completed.stdout) if completed.returncode == 0 else None
            except ValueError:
                seen = None
            problem = slot_preflight_problem(seen, slot)
            if problem:
                raise InfrastructureFailure(f"slot {slot.index} preflight failed: {problem}")
            return {"slot": slot.index, "uid": slot.uid, "gpu_uuid": slot.gpu.uuid, "ok": True}
        finally:
            shutil.rmtree(stage, ignore_errors=True)


SLOT_PROBE = r"""
import json, os, sys
expected = json.loads(sys.argv[1])
seen = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "private_listable": [], "readable_missing": []}
for path in expected["private"]:
    try:
        os.listdir(path)
        seen["private_listable"].append(path)
    except OSError:
        pass
for path in expected["readable"]:
    try:
        os.listdir(path)
    except OSError:
        seen["readable_missing"].append(path)
import torch
index = int(os.environ["QVG_DEVICE_INDEX"])
torch.cuda.set_device(index)
props = torch.cuda.get_device_properties(index)
held = torch.ones(64 * 1024 * 1024, dtype=torch.float32, device="cuda")
total = float(held.sum().item())
torch.cuda.synchronize()
seen.update(
    uid=os.getuid(), gid=os.getgid(), groups=sorted(os.getgroups()),
    current_device=int(torch.cuda.current_device()), uuid=str(getattr(props, "uuid", "")),
    sum=total,
    other_reserved_bytes=sum(int(torch.cuda.memory_reserved(j)) for j in range(torch.cuda.device_count()) if j != index),
)
print(json.dumps(seen))
"""


def slot_preflight_problem(seen: Any, slot: Slot) -> str | None:
    """Return why a slot probe result is unacceptable, or None."""
    if not isinstance(seen, dict):
        return "the probe did not complete"
    uuid = str(seen.get("uuid", ""))
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    checks = [
        (seen.get("uid") == slot.uid and seen.get("gid") == slot.uid, "wrong candidate identity"),
        (seen.get("groups") in ([], [slot.uid]), "supplementary groups survived the drop"),
        (seen.get("cuda_visible_devices") is None, "CUDA_VISIBLE_DEVICES reached the candidate"),
        (seen.get("current_device") == slot.gpu.index, "the slot did not bind its device"),
        (uuid == slot.gpu.uuid, "the slot bound a different GPU"),
        (seen.get("sum") == float(64 * 1024 * 1024), "the slot GPU computed a wrong result"),
        (seen.get("other_reserved_bytes") == 0, "the slot touched another GPU"),
        (not seen.get("private_listable"), "a controller-private path is candidate-readable"),
        (not seen.get("readable_missing"), "a model or runner path is unreadable to the candidate"),
    ]
    return next((message for ok, message in checks if not ok), None)


def supervise_root(
    jobs: Sequence[tuple[list[str], Path, int]],
    *,
    env: Mapping[str, str],
    registry: ChildRegistry,
    state: RunState | None = None,
) -> None:
    """Run trusted root children concurrently; any failure kills the others.

    Each child leads its own process group inside the controller's session,
    is killed with its group at its own deadline, and the first error, a
    timeout, or a run-wide cancel stops every sibling.
    """
    processes: list[tuple[subprocess.Popen[Any], float, str]] = []
    handles = []
    try:
        for command, log, timeout in jobs:
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w", encoding="utf-8")
            handles.append(handle)
            os.chmod(log, 0o600)
            try:
                process = subprocess.Popen(
                    command,
                    env=dict(env),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    process_group=0,
                )
            except OSError as exc:
                raise InfrastructureFailure("a trusted metric could not start") from exc
            registry.add_group(process.pid)
            processes.append((process, time.monotonic() + timeout, log.name))
        pending = list(processes)
        while pending:
            if state is not None and state.cancel.is_set():
                raise InfrastructureFailure("trusted metrics were cancelled")
            for item in list(pending):
                process, deadline, name = item
                returncode = process.poll()
                if returncode is None:
                    if time.monotonic() >= deadline:
                        raise InfrastructureFailure(f"a trusted metric exceeded its time limit ({name})")
                    continue
                if returncode:
                    raise InfrastructureFailure(f"a trusted metric returned an error ({name})")
                pending.remove(item)
            if pending:
                time.sleep(POLL_SECONDS)
    finally:
        for process, _deadline, _name in processes:
            if process.poll() is None:
                _kill_group(process)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            registry.discard(process.pid)
        for handle in handles:
            handle.close()


# --------------------------------------------------------------------------
# Sharded run
# --------------------------------------------------------------------------


def build_slots(
    release: dict[str, Any], gpus: Sequence[judge_shards.Gpu], cpus: Sequence[Sequence[int]]
) -> list[Slot]:
    uids = release["judge"]["slot_uids"]
    return [Slot(i, gpu, uids[i], tuple(cpus[i])) for i, gpu in enumerate(gpus)]


def execute_plan(
    release: dict[str, Any],
    slots: Sequence[Slot],
    plan: Sequence[Sequence[str]],
    runner: Any,
    state: RunState,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Run each slot's case list in its own thread; stop every slot on the first failure."""
    if len(plan) != len(slots):
        raise InfrastructureFailure("the shard plan does not match the slots")
    cases = {case["case_id"]: case for case in release["cases"]}
    order = {case["case_id"]: position for position, case in enumerate(release["cases"])}
    stagger = release["judge"]["start_stagger_seconds"]
    port_base = release["judge"]["master_port_base"]
    planned = [case_id for shard in plan for case_id in shard]
    if sorted(planned) != sorted(cases) or len(planned) != len(set(planned)):
        raise InfrastructureFailure("the shard plan does not cover every case exactly once")
    controls: dict[str, dict[str, Any]] = {}
    for slot, shard in zip(slots, plan):
        for case_id in shard:
            controls[case_id] = {
                "case_id": case_id,
                "model": cases[case_id]["model"],
                "order": order[case_id],
                # Unique per case and never reused within a run.
                "master_port": port_base + order[case_id],
                **slot.record(),
                "status": "not_started",
            }
    rows: dict[str, dict[str, Any]] = {}
    progress_lock = threading.Lock()

    def report() -> None:
        if progress is None:
            return
        with progress_lock:
            # Other slots mutate their own records meanwhile; a snapshot that
            # races one is simply retaken.  Diagnostics only: a failure here
            # never changes the verdict.
            for _attempt in range(3):
                try:
                    snapshot = json.loads(json.dumps(controls))
                except RuntimeError:
                    continue
                try:
                    progress({"cases": snapshot, "first_error": state.first_error})
                except Exception:
                    pass
                return

    def run_slot(position: int) -> None:
        slot = slots[position]
        cancelled = state.wait(position * stagger)
        for case_id in plan[position]:
            control = controls[case_id]
            if cancelled or state.cancel.is_set():
                control["status"] = "cancelled"
                continue
            control["started_at"] = state.now()
            control["status"] = "running"
            report()
            try:
                rows[case_id] = runner.run_case(cases[case_id], slot, control, state)
                control["status"] = "ok"
            except BaseException as exc:  # noqa: BLE001 - every failure is attributed
                kind = failure_kind(exc)
                message = _failure_message(exc)
                control["error"] = message
                first = state.fail(kind, case_id, slot.index, message)
                # Anything raised after the first failure was caused by the
                # cancel (its processes were killed), so it is not a verdict.
                control["status"] = kind if first else "cancelled"
            finally:
                control["finished_at"] = state.now()
                report()

    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(run_slot, position) for position in range(len(slots))]
        for future in futures:
            future.result()
    report()
    return controls, rows


def run_suite(
    release: dict[str, Any],
    slots: Sequence[Slot],
    plan: Sequence[Sequence[str]],
    runner: Any,
    *,
    control: Any,
    registry: ChildRegistry,
    state_factory: Callable[[Callable[[], None]], RunState] = RunState,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], RunState]:
    """Global cleanups at start, end and cancel; per-slot cleanups in between."""
    all_uids = [slot.uid for slot in slots]

    def abort() -> None:
        registry.kill_all()
        try:
            control.kill_uids(all_uids)
        except InfrastructureFailure:
            pass  # the end-of-run cleanup below still fails closed

    control.kill_uids(all_uids)
    state = state_factory(abort)
    try:
        controls, rows = execute_plan(release, slots, plan, runner, state, progress)
    finally:
        registry.kill_all()
        control.kill_uids(all_uids)
    return controls, rows, state


# --------------------------------------------------------------------------
# Aggregation and gates
# --------------------------------------------------------------------------


def case_checks(
    row: dict[str, Any], case: dict[str, Any], floor: dict[str, Any], *, stage_seconds: int
) -> dict[str, bool]:
    checks = {
        "psnr": row["fidelity"]["psnr"] >= floor["psnr_min"],
        "ssim": row["fidelity"]["ssim"] >= floor["ssim_min"],
        "lpips": row["fidelity"]["lpips"] <= floor["lpips_max"],
    }
    for dimension in VBENCH_DIMENSIONS:
        checks[f"vbench_{dimension}"] = (
            row["vbench"][dimension]
            >= case["bf16_vbench"][dimension] - floor["vbench_allowed_drop"][dimension]
        )
    # End-to-end latency stays a hard gate, against the budget this stage
    # actually ran under instead of a wall measured once on one GPU model.
    checks["wall_time"] = row["wall_seconds"] <= stage_seconds
    return checks


def aggregate(release: dict[str, Any], rows: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-model byte-sum ratios, mean gates and per-case floors.

    Independent of which slot ran a case and of how many slots there were.
    """
    expected = [case["case_id"] for case in release["cases"]]
    if len(rows) != len(expected) or set(rows) != set(expected):
        raise InfrastructureFailure("the case rows do not cover the release exactly once")
    models: dict[str, Any] = {}
    cases: dict[str, Any] = {}
    for name in MODELS:
        model = release["models"][name]
        mean_gate = model["gates"]["mean"]
        floor = model["gates"]["case_floor"]
        members = [case for case in release["cases"] if case["model"] == name]
        member_rows = [rows[case["case_id"]] for case in members]
        stored = sum(row["stored_kv_bytes"] for row in member_rows)
        bf16 = sum(row["bf16_kv_bytes"] for row in member_rows)
        mean = {key: statistics.fmean(row["fidelity"][key] for row in member_rows) for key in FIDELITY_METRICS}
        vbench = {d: statistics.fmean(row["vbench"][d] for row in member_rows) for d in VBENCH_DIMENSIONS}
        bf16_vbench = {d: statistics.fmean(case["bf16_vbench"][d] for case in members) for d in VBENCH_DIMENSIONS}
        checks = {
            "mean_psnr": mean["psnr"] >= mean_gate["psnr_min"],
            "mean_ssim": mean["ssim"] >= mean_gate["ssim_min"],
            "mean_lpips": mean["lpips"] <= mean_gate["lpips_max"],
        }
        for dimension in VBENCH_DIMENSIONS:
            checks[f"mean_vbench_{dimension}"] = (
                vbench[dimension] >= bf16_vbench[dimension] - mean_gate["vbench_allowed_drop"][dimension]
            )
        member_passed = True
        for case, row in zip(members, member_rows):
            per_case = case_checks(row, case, floor, stage_seconds=model["stage_seconds"])
            cases[case["case_id"]] = {"model": name, "checks": per_case, "passed": all(per_case.values())}
            member_passed = member_passed and all(per_case.values())
        reference = model.get("quality_reference")
        # Only a calibrating release lacks the zero point; it then scores
        # compression alone (and issues no verdict).
        quality = None if reference is None else quality_terms(
            mean,
            vbench,
            bf16_vbench,
            mean_gate["vbench_allowed_drop"],
            reference,
        )
        q = 0.0 if quality is None else quality["q"]
        ratio = bf16 / stored
        models[name] = {
            "case_count": len(members),
            "stored_kv_bytes": stored,
            "bf16_kv_bytes": bf16,
            "compression_ratio": ratio,
            "mean": {**mean, "vbench": vbench, "bf16_vbench": bf16_vbench},
            "quality": quality,
            # 1 + q > 0 whenever the mean gates pass; the clamp only keeps the
            # ungated raw score defined far past them.
            "weighted_ratio": ratio * max(1.0 + q, 0.0),
            "checks": checks,
            "passed": all(checks.values()) and member_passed,
        }
    passed = all(models[name]["passed"] for name in MODELS)
    return {
        "models": models,
        "cases": cases,
        "passed": passed,
        "score": quality_weighted_score(models),
        "compression_score": absolute_compression_score(models),
    }


def _vbench_margins(
    vbench: Mapping[str, float], bf16_vbench: Mapping[str, float], allowed_drop: Mapping[str, float]
) -> dict[str, float]:
    """1 at or above the BF16 reading, 0 at the mean gate's allowed drop, negative past it."""
    return {
        d: 1.0 - max(bf16_vbench[d] - vbench[d], 0.0) / allowed_drop[d]
        for d in VBENCH_DIMENSIONS
    }


def quality_terms(
    mean: Mapping[str, float],
    vbench: Mapping[str, float],
    bf16_vbench: Mapping[str, float],
    allowed_drop: Mapping[str, float],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """The quality term ``q`` of one model from its case means.

    ``f``: each fidelity metric's share of the way from the reference codec to
    its ideal (SSIM 1, LPIPS 0, PSNR ``PSNR_CEILING_DB``), capped at 1 and
    negative when worse than the reference; ``f`` is their mean.  ``v``: the
    tightest VBench margin to the mean gate minus the reference codec's
    tightest margin, so the dimension closest to failing decides it and
    readings above BF16 earn nothing.  ``q = 0.5 f + 0.5 v`` is 0 for the
    reference codec.
    """
    gains = {
        "psnr": min((mean["psnr"] - reference["psnr"]) / (PSNR_CEILING_DB - reference["psnr"]), 1.0),
        "ssim": min((mean["ssim"] - reference["ssim"]) / (1.0 - reference["ssim"]), 1.0),
        "lpips": min((reference["lpips"] - mean["lpips"]) / reference["lpips"], 1.0),
    }
    fidelity = statistics.fmean(gains[key] for key in FIDELITY_METRICS)
    margins = _vbench_margins(vbench, bf16_vbench, allowed_drop)
    reference_margins = _vbench_margins(reference["vbench"], bf16_vbench, allowed_drop)
    worst_dimension = min(VBENCH_DIMENSIONS, key=lambda d: margins[d])
    reference_worst = min(reference_margins.values())
    video = margins[worst_dimension] - reference_worst
    return {
        "fidelity_gain": gains,
        "f": fidelity,
        "vbench_margin": margins,
        "worst_vbench_dimension": worst_dimension,
        "worst_vbench_margin": margins[worst_dimension],
        "reference_worst_vbench_margin": reference_worst,
        "v": video,
        "q": FIDELITY_WEIGHT * fidelity + VIDEO_WEIGHT * video,
    }


def quality_weighted_score(models: Mapping[str, Mapping[str, Any]]) -> float:
    """Return the cross-model geometric mean of ``R * (1 + q)``."""

    return math.sqrt(
        models["longcat"]["weighted_ratio"]
        * models["hy_worldplay_8b"]["weighted_ratio"]
    )


def absolute_compression_score(models: Mapping[str, Mapping[str, Any]]) -> float:
    """Return the cross-model geometric mean of summed-byte ratios (informational)."""

    return math.sqrt(
        models["longcat"]["compression_ratio"]
        * models["hy_worldplay_8b"]["compression_ratio"]
    )


def finalize(
    release: dict[str, Any],
    rows: Mapping[str, dict[str, Any]],
    controls: Mapping[str, dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    summary = aggregate(release, rows)
    if not math.isfinite(summary["score"]):
        raise CandidateFailure("candidate score is invalid")
    calibrating = release["status"] == "calibrating"
    valid = summary["passed"] and not calibrating
    if calibrating:
        # Run everything and record every reading, but never issue a verdict
        # from placeholder gates: the readings calibrate the release.
        reason = "uncalibrated_release"
    else:
        reason = None if valid else "quality_gates"
    cases = {
        case["case_id"]: {
            "control": controls.get(case["case_id"]),
            "row": rows[case["case_id"]],
            **summary["cases"][case["case_id"]],
        }
        for case in release["cases"]
    }
    return {
        **context,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "outcome": "evaluated",
        "valid": bool(valid),
        "candidate_invalid": not valid and not calibrating,
        "infrastructure_invalid": calibrating,
        "reason": reason,
        "score": summary["score"] if valid else 0.0,
        "raw_score": summary["score"],
        "compression_score": summary["compression_score"],
        "gates_passed": summary["passed"],
        "objective": "quality_weighted_geometric_mean_summed_persistent_kv_compression_ratio_with_mean_and_per_case_quality_and_latency_gates",
        "models": summary["models"],
        "cases": cases,
        "first_error": None,
    }


def evaluate(
    submission: Path,
    release_path: Path,
    output: Path,
    *,
    quality_path: Path | None = None,
    expected_gpu_count: int | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = {} if context is None else context
    started = time.monotonic()
    release = load_release(release_path, quality_path=quality_path)
    context.update(
        release_status=release["status"],
        release_sha256=sha256(release_path),
        execution_budget=execution_budget(release),
    )
    if quality_path is not None:
        context.update(quality_sha256=sha256(quality_path),
                       quality_hardware=_strict_json(quality_path)["hardware"])
    require_quality(release)
    verify_frozen_files(frozen_files(release))
    prompts = prompt_lines(release)
    tools = isolation_prerequisites(release)
    gpus, fleet = evaluator_hardware(release)
    context["hardware"] = fleet
    if expected_gpu_count is not None and len(gpus) != expected_gpu_count:
        raise InfrastructureFailure("the Judge GPU count changed after the adapter derived its timeout")
    try:
        slices, cpu_mode = judge_shards.cpu_slices(gpus)
        cpus = judge_shards.slot_cpus(slices, release["judge"]["threads_per_slot"])
    except RuntimeError as exc:
        raise InfrastructureFailure(f"the Judge CPU allocation cannot host every slot: {exc}") from exc
    slots = build_slots(release, gpus, cpus)
    plan = schedule_plan(release, len(gpus))
    context.update(
        cpu_slice_mode=cpu_mode,
        slots=[slot.record() for slot in slots],
        schedule={
            "gpu_count": len(gpus),
            "plan": plan,
            "controller_budget_seconds": controller_budget_seconds(release, len(gpus)),
        },
    )
    context["source_admission"] = admit_submission(submission)

    log_root = output.parent
    registry = ChildRegistry()
    control = ProcessControl(tools.pkill)
    with tempfile.TemporaryDirectory(prefix="qvg-private-score-", dir="/tmp") as temporary:
        scratch = Path(temporary)
        os.chmod(scratch, 0o700)
        runner = CaseRunner(
            release,
            tools=tools,
            control=control,
            registry=registry,
            log_root=log_root,
            scratch=scratch,
            prompts=prompts,
        )
        control.kill_uids([slot.uid for slot in slots])
        context["slot_preflight"] = runner.preflight(slots)
        progress_path = log_root / PROGRESS_NAME
        controls, rows, state = run_suite(
            release,
            slots,
            plan,
            runner,
            control=control,
            registry=registry,
            progress=lambda snapshot: _write_json(progress_path, snapshot),
        )
    context["cases"] = {case_id: {"control": record} for case_id, record in controls.items()}
    context["schedule"]["wall_seconds"] = round(time.monotonic() - started, 3)
    if state.first_error is not None:
        context["first_error"] = state.first_error
        raise_first_error(state.first_error)
    return finalize(release, rows, controls, context)


def raise_first_error(first_error: Mapping[str, Any]) -> None:
    """The first failure decides the outcome; everything after it was the cancel."""
    if first_error["kind"] == "candidate_failure":
        raise CandidateFailure(first_error["message"])
    raise InfrastructureFailure(first_error["message"])


def failure_result(kind: str, message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    infrastructure = kind == "infrastructure_failure"
    return {
        **(context or {}),
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "outcome": kind,
        "valid": False,
        "candidate_invalid": not infrastructure,
        "infrastructure_invalid": infrastructure,
        "score": 0.0,
        "error": message,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int)
    args = parser.parse_args()
    context: dict[str, Any] = {}
    try:
        detail = evaluate(
            args.submission,
            args.release,
            args.output,
            quality_path=args.quality,
            expected_gpu_count=args.gpu_count,
            context=context,
        )
        exit_code = 0
    except CandidateFailure as exc:
        detail = failure_result("candidate_failure", str(exc), context)
        exit_code = 2
    except InfrastructureFailure as exc:
        detail = failure_result("infrastructure_failure", str(exc), context)
        exit_code = 3
    except Exception as exc:  # defensive fail-closed boundary
        detail = failure_result(
            "infrastructure_failure",
            f"unexpected protected evaluator failure: {type(exc).__name__}",
            context,
        )
        exit_code = 3
    _write_json(args.output, detail)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
