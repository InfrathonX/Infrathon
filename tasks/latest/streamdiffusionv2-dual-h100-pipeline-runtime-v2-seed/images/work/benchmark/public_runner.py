#!/usr/bin/env python3
"""Run the disclosed quick/full public suite with production worker semantics.

Seed-relative limits are measured, never read from a frozen table: the pristine
seed under /workspace/benchmark/seed_runtime is executed on *these* GPUs and its
FPS, TTFF, p99, per-rank peak memory and outputs become the local reference.
The measurement is cached per device fingerprint so iteration stays fast; the
Judge always re-measures the seed inside its own run and never trusts this cache.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/benchmark/runtime")
from evaluation import compare_saved_ledger, digest_saved_ledger
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

BENCHMARK_ROOT = Path("/workspace/benchmark")
PUBLIC_ROOT = BENCHMARK_ROOT / "public"
SEED_RUNTIME = BENCHMARK_ROOT / "seed_runtime" / "runtime_v001"
REFERENCE_DIR_NAME = "seed_reference_v2"
# Each public run writes several GB of saved frames; older runs keep only their JSON and logs.
KEEP_PUBLIC_ARTIFACTS = 2
POLL_SECONDS = 0.2
QUICK_IDS = ("public_bbb_480p_129_s2", "public_proc_square_129_s2")


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


def _gpu_rows() -> list[tuple[str, str, str, int]]:
    """(rank, uuid, name, used MiB); rank is the NVML ordinal in this container."""
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=10,
    )
    rows = []
    for line in raw.splitlines():
        if line.strip():
            index, uuid, name, used = (value.strip() for value in line.split(",", 3))
            rows.append((str(int(index)), uuid, name, int(used)))
    if len(rows) != 2 or {rank for rank, _, _, _ in rows} != set(RANK_KEYS):
        raise RuntimeError(f"expected exactly two visible GPUs at ranks {RANK_KEYS}")
    return sorted(rows)


def _gpu_snapshot() -> dict[str, int]:
    return {rank: used for rank, _, _, used in _gpu_rows()}


def _device_fingerprint() -> str:
    return "; ".join(f"{rank}:{name}:{uuid}" for rank, uuid, name, _ in _gpu_rows())


def _active_gpu_pids(uuids: set[str]) -> list[int]:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=10,
    )
    rows = []
    for line in raw.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] in uuids:
            rows.append(int(fields[1]))
    return sorted(set(rows))


def _rss_mib(process_group: int) -> float:
    total_kib = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            remainder = raw[raw.rfind(")") + 2 :].split()
            if int(remainder[2]) != process_group:
                continue
            for line in (entry / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    total_kib += int(line.split()[1])
                    break
        except (OSError, ValueError, IndexError):
            continue
    return total_kib / 1024.0


def _run_worker(candidate: Path, manifest: dict, run_root: Path, timeout: int) -> dict:
    """Execute one runtime package over a manifest with production semantics."""
    result_path = run_root / "result.json"
    status_path = run_root / "status.json"
    artifacts = run_root / "artifacts"
    staged_parent = run_root / "candidates"
    artifacts.mkdir(parents=True, exist_ok=False)
    staged_parent.mkdir(parents=True, exist_ok=True)
    staged_candidate = staged_parent / candidate.name
    shutil.copytree(candidate, staged_candidate)
    command = [
        sys.executable,
        str(BENCHMARK_ROOT / "candidate_worker.py"),
        "--candidate",
        str(staged_candidate),
        "--output",
        str(result_path),
        "--artifacts",
        str(artifacts),
        "--clip-root",
        str(PUBLIC_ROOT / "clips"),
        "--embedding-root",
        str(PUBLIC_ROOT / "embeddings"),
        "--status",
        str(status_path),
    ]
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0,1",
        PYTHONPATH=str(BENCHMARK_ROOT / "runtime"),
    )
    log_path = run_root / "worker.log"
    uuids = {rank: uuid for rank, uuid, _, _ in _gpu_rows()}
    baseline = _gpu_snapshot()
    peaks: dict[str, dict[str, int]] = {}
    rss_peaks: dict[str, float] = {}
    monitor_errors: list[str] = []
    current_case = "_startup"
    timed_out = False
    with log_path.open("wb") as worker_log:
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
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if time.monotonic() >= deadline:
                _kill_group(process)
                timed_out = True
                break
            if status_path.is_file():
                try:
                    current_case = str(json.loads(status_path.read_text(encoding="utf-8"))["case_id"])
                except (OSError, KeyError, json.JSONDecodeError):
                    pass
            try:
                sample = _gpu_snapshot()
                bucket = peaks.setdefault(current_case, dict(baseline))
                for rank, used in sample.items():
                    bucket[rank] = max(bucket.get(rank, 0), used)
                rss_peaks[current_case] = max(rss_peaks.get(current_case, 0.0), _rss_mib(process.pid))
            except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
                monitor_errors.append(f"{type(exc).__name__}: {exc}")
            time.sleep(POLL_SECONDS)
        returncode = process.wait()
    terminated, leftover = _terminate_session_group(process.pid)
    return {
        "ok": not timed_out and returncode == 0 and result_path.is_file(),
        "timed_out": timed_out,
        "returncode": returncode,
        "result": json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {},
        "artifacts": artifacts,
        "log_path": log_path,
        "baseline_memory_mib_per_rank": baseline,
        "peak_memory_mib_per_rank_per_case": peaks,
        "host_rss_growth_mib_per_case": rss_peaks,
        "post_close_memory_mib_per_rank": _gpu_snapshot(),
        "active_gpu_pids": _active_gpu_pids(set(uuids.values())),
        "terminated_session_pids": terminated,
        "leftover_session_pids": leftover,
        "monitor_errors": monitor_errors,
    }


def _prune_public_runs(build: Path, keep: int = KEEP_PUBLIC_ARTIFACTS) -> None:
    """Drop the bulky artifacts/ and candidates/ of all but the newest `keep` public runs."""
    runs = sorted(
        (path for path in build.glob("public-*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    for run_root in runs[: max(len(runs) - keep, 0)]:
        for name in ("artifacts", "candidates"):
            shutil.rmtree(run_root / name, ignore_errors=True)


def _seed_reference(manifest: dict, workspace: Path, timeout: int, mode: str) -> dict:
    """Measure the pristine seed on these GPUs, caching per device fingerprint."""
    root = workspace / "build" / REFERENCE_DIR_NAME
    index_path = root / "index.json"
    fingerprint = _device_fingerprint()
    cached: dict = {"schema_version": 1, "device_fingerprint": fingerprint, "cases": {}}
    if mode != "refresh" and index_path.is_file():
        try:
            loaded = json.loads(index_path.read_text(encoding="utf-8"))
            if loaded.get("device_fingerprint") == fingerprint and isinstance(loaded.get("cases"), dict):
                cached = loaded
        except (OSError, json.JSONDecodeError):
            pass
    wanted = [case for case in manifest["cases"] if str(case["id"]) not in cached["cases"]]
    if wanted:
        run_root = root / f"measure-{int(time.time())}-{os.getpid()}"
        try:
            outcome = _run_worker(SEED_RUNTIME, {**manifest, "cases": wanted}, run_root, timeout)
            if not outcome["ok"] or outcome["result"].get("ok") is not True:
                raise RuntimeError(
                    f"seed reference pass failed: {outcome['result'].get('error', outcome['returncode'])}"
                )
            for row in outcome["result"]["cases"]:
                case_id = str(row["id"])
                chunks = next(int(c["chunks_expected"]) for c in wanted if str(c["id"]) == case_id)
                destination = root / case_id
                shutil.rmtree(destination, ignore_errors=True)
                digest_saved_ledger(
                    outcome["artifacts"] / case_id, destination, expected_chunks=chunks
                )
                cached["cases"][case_id] = {
                    "steady_fps": positive(row["steady_fps"]),
                    "ttff_seconds": positive(row["ttff_seconds"]),
                    "push_to_emit_p99_seconds": positive(row["push_to_emit_p99_seconds"]),
                    "construction_seconds": positive(row["construction_seconds"]),
                    "peak_memory_mib_per_rank": {
                        rank: int(value)
                        for rank, value in outcome["peak_memory_mib_per_rank_per_case"]
                        .get(case_id, {})
                        .items()
                    },
                    "host_rss_growth_mib": float(
                        outcome["host_rss_growth_mib_per_case"].get(case_id, 0.0)
                    ),
                }
            cached["device_fingerprint"] = fingerprint
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(json.dumps(cached, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        finally:
            shutil.rmtree(run_root, ignore_errors=True)
    return {"root": root, "fingerprint": fingerprint, "cases": cached["cases"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--candidate-version", type=int, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--quick", action="store_true")
    selection.add_argument("--case")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--seed-reference",
        choices=("auto", "refresh"),
        default="auto",
        help="auto reuses a cached on-device seed measurement; refresh re-measures it",
    )
    args = parser.parse_args()
    if not 1 <= args.candidate_version <= 999:
        parser.error("candidate version must be in [1,999]")
    candidate = (args.workspace / "candidates" / f"runtime_v{args.candidate_version:03d}").resolve()
    try:
        admission = admit_candidate(candidate)
    except AdmissionError as exc:
        print(json.dumps({"valid": False, "score": 0.0, "failure": str(exc)}, indent=2))
        return 1
    manifest = json.loads((BENCHMARK_ROOT / "runtime" / "manifests" / "public_cases.json").read_text(encoding="utf-8"))
    available = {row["id"]: row for row in manifest["cases"]}
    if args.case:
        if args.case not in available:
            parser.error(f"unknown public case: {args.case}")
        manifest["cases"] = [available[args.case]]
    elif args.quick:
        manifest["cases"] = [available[case_id] for case_id in QUICK_IDS]
    try:
        reference = _seed_reference(manifest, args.workspace, args.timeout, args.seed_reference)
    except (OSError, RuntimeError, ValueError, KeyError, StopIteration) as exc:
        print(json.dumps({"valid": False, "score": 0.0, "failure": f"seed reference: {exc}"}, indent=2))
        return 1
    _prune_public_runs(args.workspace / "build", keep=KEEP_PUBLIC_ARTIFACTS - 1)
    run_root = args.workspace / "build" / f"public-{int(time.time())}-{os.getpid()}"
    outcome = _run_worker(candidate, manifest, run_root, args.timeout)
    if not outcome["ok"]:
        failure = "worker timeout" if outcome["timed_out"] else f"worker exit {outcome['returncode']}"
        print(
            json.dumps(
                {
                    "valid": False,
                    "score": 0.0,
                    "failure": failure,
                    "worker_log": str(outcome["log_path"]),
                    "admission": admission,
                },
                indent=2,
            )
        )
        return 1
    result = outcome["result"]
    artifacts = outcome["artifacts"]
    rows = []
    failures = []
    gates = {name: False for name in ("admission_ok", "seed_reference_ok", "worker_ok", "completeness_ok", "parity_ok", "liveness_ok", "resources_ok", "ttff_ok", "p99_ok", "construction_ok")}
    gates["admission_ok"] = True
    gates["seed_reference_ok"] = all(str(case["id"]) in reference["cases"] for case in manifest["cases"])
    gates["worker_ok"] = result.get("ok") is True
    completeness_failures = []
    parity_failures = []
    resource_failures = []
    ttff_failures = []
    p99_failures = []
    construction_failures = []
    for raw in result["cases"]:
        row = dict(raw)
        case_id = str(row["id"])
        try:
            parity = compare_saved_ledger(artifacts / case_id, reference["root"] / case_id)
            parity["reference"] = "on_device_seed_reference"
        except BaseException as exc:
            parity = {"parity_ok": False, "error": f"{type(exc).__name__}: {exc}"}
        seed = reference["cases"].get(case_id)
        row["parity"] = parity
        row["peak_memory_mib_per_rank"] = outcome["peak_memory_mib_per_rank_per_case"].get(case_id, {})
        row["host_rss_growth_mib"] = outcome["host_rss_growth_mib_per_case"].get(case_id, 0.0)
        if isinstance(seed, dict):
            try:
                row = decorate(row, seed)
                timing = timing_gates(row)
                if not timing["ttff_ok"]:
                    ttff_failures.append(case_id)
                if not timing["p99_ok"]:
                    p99_failures.append(case_id)
                if not timing["construction_ok"]:
                    construction_failures.append(case_id)
            except (ValueError, TypeError, KeyError):
                ttff_failures.append(case_id)
                p99_failures.append(case_id)
                construction_failures.append(case_id)
        else:
            ttff_failures.append(case_id)
            p99_failures.append(case_id)
            construction_failures.append(case_id)
        if parity.get("completeness_ok") is not True or parity.get("finite_ok") is not True:
            completeness_failures.append(case_id)
        if parity.get("parity_ok") is not True:
            parity_failures.append(case_id)
        if (
            not memory_ok(row["peak_memory_mib_per_rank"], seed or {})
            or float(row["host_rss_growth_mib"]) > HOST_RSS_GROWTH_CAP_MIB
        ):
            resource_failures.append(case_id)
        if case_id in completeness_failures or case_id in parity_failures:
            failures.append(case_id)
        rows.append(row)
    gates["completeness_ok"] = not completeness_failures and len(rows) == len(manifest["cases"]) and {row["id"] for row in rows} == {row["id"] for row in manifest["cases"]}
    gates["parity_ok"] = not parity_failures
    gates["liveness_ok"] = not outcome["active_gpu_pids"] and not outcome["leftover_session_pids"]
    gates["resources_ok"] = (
        not resource_failures
        and not outcome["monitor_errors"]
        and post_close_memory_ok(
            outcome["post_close_memory_mib_per_rank"], outcome["baseline_memory_mib_per_rank"]
        )
    )
    gates["ttff_ok"] = not ttff_failures
    gates["p99_ok"] = not p99_failures
    gates["construction_ok"] = not construction_failures
    valid = all(gates.values())
    output = {
        "schema_version": 1,
        "suite": "quick" if args.quick else ("one" if args.case else "full"),
        "valid": valid,
        "score": suite_metrics(rows)["score"] if valid else 0.0,
        "primary_score_name": "latency_weighted_fps",
        "primary_score_unit": "fps",
        "p99_failures": p99_failures,
        "construction_failures": construction_failures,
        "rows": rows,
        "failed_cases": failures,
        "completeness_failures": completeness_failures,
        "parity_failures": parity_failures,
        "resource_failures": resource_failures,
        "ttff_failures": ttff_failures,
        "gates": gates,
        "active_gpu_pids_after_close": outcome["active_gpu_pids"],
        "terminated_session_pids": outcome["terminated_session_pids"],
        "leftover_session_pids": outcome["leftover_session_pids"],
        "baseline_memory_mib_per_rank": outcome["baseline_memory_mib_per_rank"],
        "post_close_memory_mib_per_rank": outcome["post_close_memory_mib_per_rank"],
        "resource_monitor_errors": outcome["monitor_errors"],
        "admission": admission,
        "artifacts": str(run_root),
        "seed_reference": {
            "source": "pristine seed measured on these GPUs",
            "device_fingerprint": reference["fingerprint"],
            "cache": str(reference["root"]),
            "advisory": "the Judge re-measures the seed inside its own run and never reads this cache",
            "cases": reference["cases"],
        },
        "calibrated": gates["seed_reference_ok"],
    }
    if valid:
        output.update(suite_metrics(rows))
    (run_root / "public_result.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
