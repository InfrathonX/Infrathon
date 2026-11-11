#!/usr/bin/env python3
"""Fail-closed private verifier for the SDV2 dual-GPU pipeline task (v2).

Timing is anchored on push stamps taken by the trusted controller, the steady
window runs from the first steady push to the last emission, the GPU memory
cap is the same on both ranks, and construction time is gated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pwd
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, "/opt/sdv2_judge/worker_runtime")
from evaluation import compare_saved_ledger, digest_saved_ledger
from hardware import MIN_COMPUTE_CAPABILITY, MIN_DEVICE_MEMORY_BYTES, MIN_HOST_MEMORY_BYTES
from scoring import (
    HOST_RSS_GROWTH_CAP_MIB,
    RANK_KEYS,
    decorate,
    memory_ok,
    positive,
    post_close_memory_ok,
    suite_metrics,
    timing_gates,
)
from source_admission import AdmissionError, admit_candidate

GATE_NAMES = (
    "profile_ok",
    "hashes_ok",
    "admission_ok",
    "seed_reference_ok",
    "worker_ok",
    "completeness_ok",
    "parity_ok",
    "liveness_ok",
    "resources_ok",
    "ttff_ok",
    "p99_ok",
    "construction_ok",
)
APP_ROOT = Path("/opt/sdv2_judge")
PRIVATE_ROOT = APP_ROOT / "private"
FIXTURE_ROOT = APP_ROOT / "fixtures"
WORKER_RUNTIME = APP_ROOT / "worker_runtime"
PROTECTED_MANIFEST = APP_ROOT / "protected_hashes.txt"
# Pristine seed shipped in this image and hash-protected.  It is re-measured on
# the evaluation devices in every run; it is the only source of seed FPS, TTFF,
# p99, per-rank peak memory and reference outputs.
SEED_RUNTIME = APP_ROOT / "seed_runtime" / "runtime_v001"
WORKER_TIMEOUT = int(os.environ.get("SDV2_WORKER_TIMEOUT", "1800"))
JUDGE_TIMEOUT = 4200
FORMAL_CASE_COUNT = 12
POLL_SECONDS = 0.2
EVALUATION_CONTEXT = Path("/tmp/.sforge-evaluation-context.json")
EVALUATION_PROTOCOL_ID = "sdv2-pipeline-v2-full-final"
FORMAL_SUITE_ID = "sdv2-pipeline-final-v2"


class VerificationError(RuntimeError):
    pass


class SeedReferenceError(RuntimeError):
    """The trusted in-run seed reference failed; the candidate is not at fault."""


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path, limit: int) -> dict:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > limit:
        raise VerificationError(f"missing, linked, or oversized JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"JSON root must be an object: {path}")
    return value


def load_profile(app_dir: Path) -> dict:
    # SForge installs authoritative round/version metadata through a root-only
    # private file.  Never trust candidate-writable workspace metadata here.
    path = Path(os.environ.get("SDV2_PROFILE_FILE", EVALUATION_CONTEXT))
    profile = _read_json(path, 64 * 1024)
    round_id = profile.get("round_id")
    _suite_for_round(round_id)
    version = profile.get("candidate_version")
    if round_id.startswith(("agent-", "auto-")) and "candidate_version" in profile:
        raise VerificationError("interactive rounds must select FINAL_VERSION without an override")
    if version is not None and (
        isinstance(version, bool) or not isinstance(version, int) or not 1 <= version <= 999
    ):
        raise VerificationError("candidate_version must be an integer in [1,999]")
    return profile


def _suite_for_round(round_id: str) -> str:
    if isinstance(round_id, str) and (
        round_id == "final"
        or re.fullmatch(r"(?:agent|auto)-[1-9][0-9]*", round_id)
        or re.fullmatch(r"(?:manual|candidate)-[A-Za-z0-9][A-Za-z0-9_.-]*", round_id)
    ):
        return "final"
    raise VerificationError("round id has no suite mapping")


def _formal_suite() -> tuple[dict, dict]:
    """All formal rounds use the same hash-bound complete private suite."""
    policy = _read_json(WORKER_RUNTIME / "evaluation_protocol.json", 64 * 1024)
    if (
        policy.get("protocol_id") != EVALUATION_PROTOCOL_ID
        or policy.get("suite_id") != FORMAL_SUITE_ID
        or policy.get("case_count") != FORMAL_CASE_COUNT
        or policy.get("timeouts") != {
            "worker_pass_seconds": WORKER_TIMEOUT,
            "judge_controller_seconds": JUDGE_TIMEOUT,
            "outer_judge_seconds": JUDGE_TIMEOUT,
        }
        or os.environ.get("SDV2_TOTAL_TIMEOUT", str(JUDGE_TIMEOUT)) != str(JUDGE_TIMEOUT)
    ):
        raise VerificationError("formal evaluation protocol mismatch")
    path = PRIVATE_ROOT / "final_cases.json"
    manifest = _read_json(path, 2 * 1024 * 1024)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != policy.get("selected_manifest_sha256"):
        raise VerificationError("formal suite manifest hash mismatch")
    cases = manifest.get("cases")
    if manifest.get("suite") != FORMAL_SUITE_ID or not isinstance(cases, list) or len(cases) != FORMAL_CASE_COUNT:
        raise VerificationError("formal suite is incomplete")
    return manifest, {
        "evaluation_protocol_id": EVALUATION_PROTOCOL_ID,
        "suite_id": FORMAL_SUITE_ID,
        "suite_manifest_sha256": digest,
        "formal_case_count": len(cases),
    }


def verify_protected_hashes() -> dict[str, str]:
    roots = {
        "judge": APP_ROOT,
        "worker": WORKER_RUNTIME,
        "seed": SEED_RUNTIME.parent,
        "private": PRIVATE_ROOT,
        "fixture": FIXTURE_ROOT,
    }
    if not PROTECTED_MANIFEST.is_file() or PROTECTED_MANIFEST.is_symlink():
        raise VerificationError("protected hash manifest is missing or linked")
    seen: dict[str, str] = {}
    for number, raw in enumerate(PROTECTED_MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        digest, separator, label = raw.partition(" ")
        namespace, colon, relative = label.strip().partition(":")
        if len(digest) != 64 or not separator or not colon or namespace not in roots:
            raise VerificationError(f"malformed protected hash line {number}")
        path = roots[namespace] / relative
        if not path.is_file() or path.is_symlink():
            raise VerificationError(f"protected file missing or linked: {label}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise VerificationError(f"protected hash mismatch: {label}")
        seen[label.strip()] = actual
    if not seen:
        raise VerificationError("protected hash manifest is empty")
    return seen


def _candidate_dir(app_dir: Path, profile: dict) -> Path:
    workspace = app_dir / "workdir" / "sdv2_pipeline_lab"
    version = profile.get("candidate_version")
    if version is None:
        selector = workspace / "FINAL_VERSION"
        try:
            text = selector.read_text(encoding="utf-8").strip()
            if not text.isdigit():
                raise ValueError
            version = int(text)
        except (OSError, ValueError) as exc:
            raise VerificationError("invalid FINAL_VERSION") from exc
    if not 1 <= int(version) <= 999:
        raise VerificationError("selected candidate version is outside [1,999]")
    candidate = workspace / "candidates" / f"runtime_v{int(version):03d}"
    resolved = candidate.resolve()
    if resolved.parent != (workspace / "candidates").resolve():
        raise VerificationError("candidate path escaped candidates root")
    return resolved


def _kill_group(process: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def _session_pids(process_group: int) -> list[int]:
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            remainder = raw[raw.rfind(")") + 2 :].split()
            # A container without an init reaper can retain orphaned zombie
            # compiler helpers.  Zombies cannot execute or own CUDA state and
            # therefore are not live worker children for this gate.
            if remainder[0] != "Z" and int(remainder[2]) == process_group:
                matches.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return sorted(matches)


def _rss_mib(process_group: int) -> float:
    total_kib = 0
    for pid in _session_pids(process_group):
        try:
            for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    total_kib += int(line.split()[1])
                    break
        except (OSError, ValueError, IndexError):
            continue
    return total_kib / 1024.0


def _terminate_session_group(process_group: int) -> tuple[list[int], list[int]]:
    """Reap helpers that outlive an otherwise clean worker parent exit."""
    initial = _session_pids(process_group)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not _session_pids(process_group):
            break
        try:
            os.killpg(process_group, sig)
        except ProcessLookupError:
            break
        for _ in range(50):
            if not _session_pids(process_group):
                break
            time.sleep(0.1)
    return initial, _session_pids(process_group)


def _gpu_rows() -> list[tuple[str, str, int]]:
    """(rank, uuid, used MiB) for the two devices this container can see.

    The rank is the NVML ordinal inside this container, which is stable for the
    whole evaluation.  Memory evidence is therefore keyed by rank and compared
    only against a reference measured in the same run on the same devices; the
    UUID travels beside it as provenance.
    """
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=10,
    )
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        index, uuid, used = (field.strip() for field in line.split(",", 2))
        rows.append((str(int(index)), uuid, int(used)))
    if len(rows) != 2 or {rank for rank, _, _ in rows} != set(RANK_KEYS):
        raise VerificationError(f"expected exactly two visible GPUs at ranks {RANK_KEYS}")
    return sorted(rows)


def _gpu_snapshot() -> dict[str, int]:
    return {rank: used for rank, _, used in _gpu_rows()}


def _gpu_uuids() -> dict[str, str]:
    return {rank: uuid for rank, uuid, _ in _gpu_rows()}


def _active_gpu_pids(uuids: set[str]) -> list[int]:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=10,
    )
    pids = []
    for line in raw.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] in uuids:
            pids.append(int(fields[1]))
    return sorted(set(pids))


def _safe_copy_inputs(manifest: dict, inputs: Path) -> tuple[Path, Path]:
    clips = inputs / "clips"
    embeddings = inputs / "embeddings"
    clips.mkdir()
    embeddings.mkdir()
    copied: set[Path] = set()
    for case in manifest["cases"]:
        prompt_id = str(case["prompt_id"])
        prompt_source = (
            PRIVATE_ROOT / "embeddings" / "hidden" / f"{prompt_id}.pt"
            if case["tier"] == "final"
            else FIXTURE_ROOT / "embeddings" / "public" / f"{prompt_id}.pt"
        )
        targets = [(prompt_source, embeddings / f"{prompt_id}.pt")]
        if case["input"]["kind"] == "asset":
            asset_id = str(case["input"]["asset_id"])
            targets.append((PRIVATE_ROOT / "clips" / f"{asset_id}.npy", clips / f"{asset_id}.npy"))
        for source, target in targets:
            if target in copied:
                continue
            if not source.is_file() or source.is_symlink():
                raise VerificationError(f"case fixture is missing or linked: {source.name}")
            shutil.copyfile(source, target)
            copied.add(target)
    return clips, embeddings


def _spawn_worker(candidate: Path, manifest: dict, log_dir: Path, label: str = "candidate") -> tuple[dict, Path]:
    worker = APP_ROOT / "candidate_worker.py"
    job = Path(tempfile.mkdtemp(prefix=f"sdv2-{label}-", dir="/tmp"))
    artifacts = job / "artifacts"
    inputs = job / "inputs"
    staged_parent = job / "candidates"
    artifacts.mkdir()
    inputs.mkdir()
    staged_parent.mkdir()
    staged_candidate = staged_parent / candidate.name
    shutil.copytree(candidate, staged_candidate)
    clip_root, embedding_root = _safe_copy_inputs(manifest, inputs)
    result_path = job / "result.json"
    status_path = job / "status.json"
    if os.geteuid() == 0:
        account = pwd.getpwnam("agent")
        for path in (job, artifacts, inputs, clips := clip_root, embeddings := embedding_root, staged_parent):
            os.chown(path, account.pw_uid, account.pw_gid)
            path.chmod(0o700)
        for root in (staged_candidate, clips, embeddings):
            for path in (root, *root.rglob("*")):
                os.chown(path, account.pw_uid, account.pw_gid)
                path.chmod(0o700 if path.is_dir() else 0o400)
        command = ["setpriv", "--reuid=agent", "--regid=agent", "--init-groups"]
    else:
        command = []
    command += [
        sys.executable,
        str(worker),
        "--candidate",
        str(staged_candidate),
        "--output",
        str(result_path),
        "--artifacts",
        str(artifacts),
        "--clip-root",
        str(clip_root),
        "--embedding-root",
        str(embedding_root),
        "--status",
        str(status_path),
    ]
    environment = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": "/home/agent",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES", "all"),
        "PYTHONPATH": str(WORKER_RUNTIME),
        "SDV2_ASSETS_ROOT": "/opt/sdv2_weights",
        "STREAMDIFFUSIONV2_ROOT": "/opt/sdv2_weights",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    worker_log = (log_dir / f"{label}_worker.log").open("ab")
    gpu_uuids = _gpu_uuids()
    baseline_gpu = _gpu_snapshot()
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=worker_log,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )
    assert process.stdin is not None
    process.stdin.write(json.dumps(manifest, separators=(",", ":")).encode())
    process.stdin.close()
    deadline = time.monotonic() + WORKER_TIMEOUT
    peak_gpu: dict[str, dict[str, int]] = {}
    peak_rss: dict[str, float] = {}
    monitor_errors: list[str] = []
    current_case = "_startup"
    while process.poll() is None:
        if time.monotonic() >= deadline:
            _kill_group(process)
            worker_log.close()
            raise VerificationError(f"{label} worker timed out")
        if status_path.is_file():
            try:
                status = _read_json(status_path, 16 * 1024)
                if isinstance(status.get("case_id"), str):
                    current_case = status["case_id"]
            except VerificationError:
                pass
        try:
            snapshot = _gpu_snapshot()
            bucket = peak_gpu.setdefault(current_case, dict(baseline_gpu))
            for rank, used in snapshot.items():
                bucket[rank] = max(bucket.get(rank, 0), used)
            peak_rss[current_case] = max(peak_rss.get(current_case, 0.0), _rss_mib(process.pid))
        except (OSError, subprocess.SubprocessError, VerificationError, ValueError) as exc:
            monitor_errors.append(f"{type(exc).__name__}: {exc}")
        time.sleep(POLL_SECONDS)
    worker_log.close()
    returncode = process.wait()
    terminated_helpers, leftover = _terminate_session_group(process.pid)
    visible_uuids = set(gpu_uuids.values())
    idle = False
    post_close_gpu = {}
    active_gpu_pids: list[int] = []
    for _ in range(150):
        try:
            post_close_gpu = _gpu_snapshot()
            active_gpu_pids = _active_gpu_pids(visible_uuids)
            if not active_gpu_pids:
                idle = True
                break
        except (OSError, subprocess.SubprocessError, VerificationError, ValueError) as exc:
            monitor_errors.append(f"{type(exc).__name__}: {exc}")
        time.sleep(POLL_SECONDS)
    worker_result = _read_json(result_path, 16 * 1024 * 1024) if result_path.is_file() else {}
    if returncode != 0 or worker_result.get("ok") is not True:
        error = worker_result.get("error", "worker produced no result")
        raise VerificationError(f"{label} worker exited {returncode}: {error}")
    worker_result["resources"] = {
        "baseline_memory_mib_per_rank": baseline_gpu,
        "peak_memory_mib_per_case": peak_gpu,
        "host_rss_growth_mib_per_case": peak_rss,
        "post_close_memory_mib_per_rank": post_close_gpu,
        "gpu_uuid_per_rank": gpu_uuids,
        "monitor_errors": monitor_errors,
    }
    worker_result["liveness"] = {
        "clean_exit": returncode == 0,
        "terminated_session_pids": terminated_helpers,
        "leftover_session_pids": leftover,
        "active_gpu_pids": active_gpu_pids,
        "gpus_idle": idle,
    }
    for row in worker_result["cases"]:
        case_id = str(row["id"])
        row["peak_memory_mib_per_rank"] = peak_gpu.get(case_id, {})
        row["host_rss_growth_mib"] = peak_rss.get(case_id, 0.0)
    return worker_result, job


def _golden_root(suite: str) -> Path:
    return PRIVATE_ROOT / "goldens" / suite


def _check_case_rows(manifest: dict, worker: dict, what: str) -> dict[str, int]:
    expected = {str(case["id"]): int(case["chunks_expected"]) for case in manifest["cases"]}
    rows = worker.get("cases")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise VerificationError(f"{what} case rows are missing or duplicated")
    if {str(row.get("id")) for row in rows} != set(expected):
        raise VerificationError(f"{what} case set differs from the selected suite")
    for row in rows:
        case_id = str(row["id"])
        wanted = expected[case_id]
        if (
            row.get("chunks_expected") != wanted
            or row.get("frame_chunks") != wanted
            or row.get("latent_chunks") != wanted
        ):
            raise VerificationError(f"incomplete or implausible output ledger for {case_id}")
    return expected


def _compare_outputs(
    manifest: dict, worker: dict, suite: str, job: Path, reference_root: Path
) -> dict[str, dict]:
    """Compare the candidate against the seed outputs produced in this same run.

    Both runs used the same devices, drivers and pinned software, so bit-exact
    equality is a statement about the candidate's arithmetic, not about the GPU
    model.  The frozen H100 goldens are still compared when present, but only as
    a recorded diagnostic that never gates.
    """
    expected = _check_case_rows(manifest, worker, "worker")
    comparisons = {}
    for case_id, chunks in expected.items():
        candidate_dir = job / "artifacts" / case_id
        try:
            comparison = compare_saved_ledger(candidate_dir, reference_root / case_id)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise VerificationError(f"cannot validate output ledger for {case_id}: {exc}") from exc
        comparison["reference"] = "in_run_seed_reference"
        comparison["reference_chunks"] = chunks
        golden_dir = _golden_root(suite) / case_id
        if golden_dir.is_dir():
            try:
                diagnostic = compare_saved_ledger(candidate_dir, golden_dir)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                diagnostic = {"error": f"{type(exc).__name__}: {exc}"}
            comparison["frozen_golden_diagnostic"] = diagnostic
        comparisons[case_id] = comparison
    return comparisons


def _validate_completeness(comparisons: dict[str, dict]) -> None:
    failures = [
        case_id
        for case_id, row in comparisons.items()
        if row.get("completeness_ok") is not True or row.get("finite_ok") is not True
    ]
    if failures:
        raise VerificationError("incomplete, unordered, duplicate, or non-finite cases: " + ", ".join(failures))


def _validate_parity(comparisons: dict[str, dict]) -> None:
    failures = [case_id for case_id, row in comparisons.items() if row.get("parity_ok") is not True]
    if failures:
        raise VerificationError("bit-exact parity failed: " + ", ".join(failures))


def _validate_liveness(worker: dict) -> None:
    row = worker.get("liveness")
    if (
        not isinstance(row, dict)
        or row.get("clean_exit") is not True
        or row.get("leftover_session_pids") != []
        or row.get("active_gpu_pids") != []
        or row.get("gpus_idle") is not True
    ):
        raise VerificationError(
            "worker did not exit cleanly or release both GPUs: "
            + json.dumps(row, sort_keys=True)
        )


def _validate_hardware_record(result: Mapping[str, Any]) -> dict:
    """Capability envelope check on the record the trusted worker reported."""
    hardware = result.get("hardware")
    devices = hardware.get("devices") if isinstance(hardware, dict) else None
    if not isinstance(devices, list) or len(devices) != len(RANK_KEYS) or hardware.get("count") != len(RANK_KEYS):
        raise VerificationError("capability envelope violated: two CUDA devices are required")
    for device in devices:
        capability = device.get("capability") if isinstance(device, dict) else None
        memory = device.get("total_memory_bytes") if isinstance(device, dict) else None
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or tuple(int(value) for value in capability) < MIN_COMPUTE_CAPABILITY
            or isinstance(memory, bool)
            or not isinstance(memory, int)
            or memory < MIN_DEVICE_MEMORY_BYTES
        ):
            raise VerificationError(f"capability envelope violated by device record: {device}")
    host_memory = hardware.get("host_memory_bytes")
    if isinstance(host_memory, bool) or not isinstance(host_memory, int) or host_memory < MIN_HOST_MEMORY_BYTES:
        raise VerificationError("capability envelope violated: host memory below the task minimum")
    return hardware


def _seed_reference(manifest: dict, log_dir: Path) -> tuple[dict, Path, Path]:
    """Measure the pristine seed on this run's devices and keep its outputs.

    Everything the candidate is later judged against - seed FPS, TTFF, p99,
    per-rank peak memory and the reference frames/latents - comes from here.
    A failure in this pass is a harness failure, not a candidate failure.
    """
    job = None
    reference_root = None
    try:
        admit_candidate(SEED_RUNTIME)
        worker, job = _spawn_worker(SEED_RUNTIME, manifest, log_dir, label="seed")
        hardware = _validate_hardware_record(worker)
        _validate_liveness(worker)
        _check_case_rows(manifest, worker, "seed reference")
        resources = worker.get("resources")
        if not isinstance(resources, dict) or resources.get("monitor_errors"):
            raise VerificationError("seed reference resource monitor reported errors")
        reference_root = Path(tempfile.mkdtemp(prefix="sdv2-reference-", dir="/tmp"))
        chunks = {str(case["id"]): int(case["chunks_expected"]) for case in manifest["cases"]}
        cases = {}
        for row in worker["cases"]:
            case_id = str(row["id"])
            peaks = row.get("peak_memory_mib_per_rank")
            if not isinstance(peaks, dict) or set(peaks) != set(RANK_KEYS):
                raise VerificationError(f"seed reference memory evidence is missing for {case_id}")
            digest_saved_ledger(
                job / "artifacts" / case_id,
                reference_root / case_id,
                expected_chunks=chunks[case_id],
            )
            cases[case_id] = {
                "steady_fps": positive(row["steady_fps"]),
                "ttff_seconds": positive(row["ttff_seconds"]),
                "push_to_emit_p99_seconds": positive(row["push_to_emit_p99_seconds"]),
                "construction_seconds": positive(row["construction_seconds"]),
                "peak_memory_mib_per_rank": {rank: int(peaks[rank]) for rank in RANK_KEYS},
                "host_rss_growth_mib": float(row.get("host_rss_growth_mib", 0.0)),
            }
        reference = {
            "schema_version": 1,
            "source": "pristine seed runtime re-measured on the evaluation devices in this run",
            "runtime": SEED_RUNTIME.name,
            "hardware": hardware,
            "gpu_uuid_per_rank": resources.get("gpu_uuid_per_rank"),
            "baseline_memory_mib_per_rank": resources.get("baseline_memory_mib_per_rank"),
            "post_close_memory_mib_per_rank": resources.get("post_close_memory_mib_per_rank"),
            "cases": cases,
        }
        return reference, reference_root, job
    except BaseException as exc:
        for path in (job, reference_root):
            if path is not None:
                shutil.rmtree(path, ignore_errors=True)
        if isinstance(exc, (VerificationError, AdmissionError, OSError, ValueError, KeyError, json.JSONDecodeError)):
            raise SeedReferenceError(f"seed reference pass failed: {exc}") from exc
        raise


def _validate_resources(worker: dict, reference: dict) -> None:
    resources = worker.get("resources")
    if not isinstance(resources, dict):
        raise VerificationError("resource evidence is missing")
    if resources.get("monitor_errors"):
        raise VerificationError("resource monitor reported errors")
    for row in worker["cases"]:
        peaks = row.get("peak_memory_mib_per_rank")
        if not memory_ok(peaks, reference["cases"].get(str(row["id"]), {})):
            raise VerificationError(f"GPU memory cap exceeded or missing for {row['id']}")
        if positive(row.get("host_rss_growth_mib", math.inf)) > HOST_RSS_GROWTH_CAP_MIB:
            raise VerificationError(f"host RSS cap exceeded for {row['id']}")
    if not post_close_memory_ok(
        resources.get("post_close_memory_mib_per_rank"),
        resources.get("baseline_memory_mib_per_rank"),
    ):
        raise VerificationError("post-close GPU memory exceeded this run's own pre-run occupancy")


def _decorate_and_validate_ttff(
    worker: dict,
    comparisons: dict[str, dict],
    reference: dict,
) -> list[dict]:
    reference_cases = reference["cases"]
    rows = []
    for raw in worker["cases"]:
        row = dict(raw)
        case_id = str(row["id"])
        baseline = reference_cases.get(case_id)
        if not isinstance(baseline, dict):
            raise VerificationError(f"seed reference is missing {case_id}")
        row = decorate(row, baseline)
        if not timing_gates(row)["ttff_ok"]:
            raise VerificationError(f"TTFF exceeded {row['ttff_limit_seconds']:.6f}s for {case_id}")
        row["parity"] = comparisons[case_id]
        rows.append(row)
    return rows


def _score(rows: list[dict]) -> float:
    return suite_metrics(rows)["score"]


def _validate_p99(rows):
    for row in rows:
        if not timing_gates(row)["p99_ok"]:
            raise VerificationError(f"p99 exceeded {row['p99_limit_seconds']:.6f}s for {row['id']}")


def _validate_construction(rows):
    for row in rows:
        if not timing_gates(row)["construction_ok"]:
            raise VerificationError(
                f"construction exceeded {row['construction_limit_seconds']:.3f}s for {row['id']}"
            )


def verify(app_dir: Path, log_dir: Path) -> dict:
    started = time.monotonic()
    result = {
        "schema_version": 1,
        "task": "streamdiffusionv2_dual_h100_pipeline_runtime_seed_v2",
        "valid": False,
        "score": 0.0,
        "gates": {name: False for name in GATE_NAMES},
        "failure_detail": None,
        "trusted_failure": False,
        "rows": [],
    }
    job = None
    seed_job = None
    reference_root = None
    try:
        profile = load_profile(app_dir)
        result["profile"] = profile
        result["gates"]["profile_ok"] = True
        verify_protected_hashes()
        result["gates"]["hashes_ok"] = True
        candidate = _candidate_dir(app_dir, profile)
        try:
            result["admission"] = admit_candidate(candidate)
        except AdmissionError as exc:
            raise VerificationError(str(exc)) from exc
        result["gates"]["admission_ok"] = True
        suite = _suite_for_round(str(profile["round_id"]))
        result["suite"] = suite
        manifest, protocol = _formal_suite()
        result.update(protocol)
        # Trusted reference pass: the pristine seed on this run's own devices.
        reference, reference_root, seed_job = _seed_reference(manifest, log_dir)
        shutil.rmtree(seed_job, ignore_errors=True)
        seed_job = None
        result["seed_reference"] = reference
        result["hardware"] = reference["hardware"]
        # `calibrated` now means "this run measured its own seed reference".
        result["calibrated"] = True
        result["gates"]["seed_reference_ok"] = True
        worker, job = _spawn_worker(candidate, manifest, log_dir)
        result["gates"]["worker_ok"] = True
        # Preserve private diagnostic evidence even if a later gate rejects.
        # The public structured adapter suppresses these rows for invalid runs.
        result["rows"] = worker.get("cases", [])
        result["liveness"] = worker.get("liveness")
        result["resources"] = worker.get("resources")
        _validate_hardware_record(worker)
        comparisons = _compare_outputs(manifest, worker, suite, job, reference_root)
        result["comparisons"] = comparisons
        _validate_completeness(comparisons)
        result["gates"]["completeness_ok"] = True
        _validate_parity(comparisons)
        result["gates"]["parity_ok"] = True
        _validate_liveness(worker)
        result["gates"]["liveness_ok"] = True
        _validate_resources(worker, reference)
        result["gates"]["resources_ok"] = True
        rows = _decorate_and_validate_ttff(worker, comparisons, reference)
        result["gates"]["ttff_ok"] = True
        _validate_p99(rows)
        result["gates"]["p99_ok"] = True
        _validate_construction(rows)
        result["gates"]["construction_ok"] = True
        if not all(result["gates"].values()):
            raise VerificationError("one or more gates remained false")
        result["rows"] = rows
        result.update(suite_metrics(rows))
        result["valid"] = True
    except SeedReferenceError as exc:
        result["failure_detail"] = f"{type(exc).__name__}: {exc}"
        result["trusted_failure"] = True
        result["valid"] = False
        result["score"] = 0.0
    except BaseException as exc:
        result["failure_detail"] = f"{type(exc).__name__}: {exc}"
        result["valid"] = False
        result["score"] = 0.0
    finally:
        result["verifier_wall_seconds"] = time.monotonic() - started
        for path in (job, seed_job, reference_root):
            if path is not None:
                shutil.rmtree(path, ignore_errors=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, default=Path("/app"))
    parser.add_argument("--log-dir", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()
    result = verify(args.app_dir, args.log_dir)
    _atomic_json(args.log_dir / "sdv2_pipeline_v2_results.json", result)
    _atomic_json(
        args.log_dir / "reward.json",
        {
            "reward": result["score"],
            "valid": result["valid"],
            "gates": result["gates"],
            "trusted_failure": bool(result.get("trusted_failure")),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
