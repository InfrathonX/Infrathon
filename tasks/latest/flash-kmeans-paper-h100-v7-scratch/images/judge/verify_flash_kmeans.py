#!/usr/bin/env python3
"""Private H100 verifier for the K-means kernel optimization task."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pwd
import re
import shutil
import stat
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
_JUDGE_BENCHMARK_DIR = Path("/opt/flash_kmeans_benchmark")
if not _JUDGE_BENCHMARK_DIR.is_dir():
    _JUDGE_BENCHMARK_DIR = Path(__file__).resolve().parent / "benchmark"
sys.path.insert(0, str(_JUDGE_BENCHMARK_DIR))

import torch

from benchmark_cases import (
    ASSIGN_WEIGHT,
    DEPLOYMENT_WEIGHT,
    FULL_WEIGHT,
    RUNTIME_WEIGHT,
    SCORE_GROUP_WEIGHTS,
    SCORE_PROFILE,
    UPDATE_WEIGHT,
)
from deployment_bench import run_deployment_case
from kernel_bench import (
    CaseSpec,
    DEVICE_SYNCHRONIZE,
    EMPTY_CACHE,
    MAX_MEMORY_ALLOCATED,
    MEMORY_ALLOCATED,
    RESET_PEAK_MEMORY,
    benchmark_stable_latency_ms,
    check_output,
    clone_guards,
    inputs_unchanged,
    load_candidate,
    make_inputs,
    memory_diagnostics,
    memory_limit_bytes,
    benchmark_case_to_spec,
    peak_output,
    run_stage,
)
from hidden_cases import (
    COARSE_DEPLOYMENT_FAMILIES,
    HIDDEN_DEPLOYMENT_CASES,
    HIDDEN_RUNTIME_CASES,
)
from source_admission import scan_submission


TRUSTED_EXP = math.exp
TRUSTED_LOG = math.log
RESULT_NAMES = ("reward.json", "reward.txt", "flash_kmeans_results.json")
FIRST_RUN_WEIGHT = 0.0
REQUIRED_PROTECTED_PATHS = frozenset(
    {
        "/opt/flash_kmeans_benchmark/kernel_bench.py",
        "/opt/flash_kmeans_benchmark/deployment_bench.py",
        "/opt/flash_kmeans_benchmark/benchmark_cases.py",
        "/opt/flash_kmeans_benchmark/source_admission.py",
        "/app/workdir/flash_kmeans_lab/AGENT_TASK.md",
        "/app/workdir/flash_kmeans_lab/README_AGENT.md",
        "/opt/flash_kmeans_judge/verify_flash_kmeans.py",
        "/opt/flash_kmeans_judge/run_judge.py",
        "/opt/flash_kmeans_judge/hidden_cases.py",
    }
)


class ResultFiles:
    """Pre-open result descriptors while root; usable after permanent setuid."""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        log_dir.chmod(0o700)
        self.handles: dict[str, TextIO] = {}
        for name in RESULT_NAMES:
            path = log_dir / name
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink()
            except OSError:
                pass
            self.handles[name] = path.open("w", encoding="utf-8")

    def write(self, detail: dict[str, Any], reward: float) -> None:
        payload = dict(detail)
        payload["reward"] = float(reward)
        numeric: dict[str, float] = {"reward": float(reward)}
        for key, value in payload.items():
            if isinstance(value, bool):
                numeric[key] = 1.0 if value else 0.0
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                numeric[key] = float(value)
        texts = {
            "reward.json": json.dumps(numeric, indent=2, sort_keys=True, allow_nan=False) + "\n",
            "reward.txt": f"{float(reward)}\n",
            "flash_kmeans_results.json": (
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            ),
        }
        for name, text in texts.items():
            handle = self.handles[name]
            handle.seek(0)
            handle.truncate()
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())


def suppress_process_output() -> None:
    """FD-level suppression: candidate/native writes cannot forge a marker."""
    sink = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(sink, 1)
        os.dup2(sink, 2)
    finally:
        if sink > 2:
            os.close(sink)


def hidden_runtime_specs(suite: str) -> list[CaseSpec]:
    if suite == "smoke":
        return [
            CaseSpec(
                "hidden_smoke_assign",
                "assign",
                "close_centroid",
                1,
                1024,
                32,
                32,
                81001,
                weight=ASSIGN_WEIGHT,
                score_group="assign",
                component="assignment",
                measurement_mode="steady_state",
                verification="sampled_exact",
            ),
            CaseSpec(
                "hidden_smoke_update",
                "update",
                "online",
                1,
                2048,
                32,
                32,
                81002,
                "hot99_empty",
                weight=UPDATE_WEIGHT,
                score_group="update",
                component="centroid_update",
                measurement_mode="steady_state",
                verification="sampled_exact",
            ),
            CaseSpec(
                "hidden_smoke_full",
                "full",
                "online",
                1,
                1024,
                32,
                32,
                81003,
                weight=FULL_WEIGHT,
                score_group="full",
                component="combined_iteration",
                measurement_mode="steady_state",
                verification="sampled_exact",
            ),
            CaseSpec(
                "hidden_smoke_tie_gate",
                "assign",
                "online_tie",
                2,
                1031,
                33,
                31,
                81004,
                tie=True,
                weight=0.0,
                role="correctness",
                component="assignment",
                measurement_mode="steady_state",
            ),
        ]
    runtime = [
        benchmark_case_to_spec(case, seed=81_000 + index)
        for index, case in enumerate(HIDDEN_RUNTIME_CASES)
    ]
    gates = [
        CaseSpec(
            "hidden_exact_tie_tile_boundaries",
            "assign",
            "online_tie",
            2,
            4099,
            257,
            33,
            82_000,
            tie=True,
            weight=0.0,
            warmup=0,
            role="correctness",
            component="assignment",
            measurement_mode="steady_state",
            verification="full_exact",
            visibility="hidden",
        ),
        CaseSpec(
            "hidden_close_centroids",
            "assign",
            "close_centroid",
            1,
            8191,
            127,
            128,
            82_001,
            weight=0.0,
            warmup=0,
            role="correctness",
            component="assignment",
            measurement_mode="steady_state",
            verification="full_gap_aware",
            visibility="hidden",
        ),
        CaseSpec(
            "hidden_non_power_of_two_full",
            "full",
            "online",
            3,
            65_537,
            65,
            127,
            82_002,
            weight=0.0,
            warmup=0,
            role="correctness",
            component="combined_iteration",
            measurement_mode="steady_state",
            visibility="hidden",
        ),
        CaseSpec(
            "hidden_single_cluster_full",
            "full",
            "online",
            2,
            4097,
            1,
            7,
            82_003,
            weight=0.0,
            warmup=0,
            role="correctness",
            component="combined_iteration",
            measurement_mode="steady_state",
            visibility="hidden",
        ),
        CaseSpec(
            "hidden_hot_empty_update",
            "update",
            "online",
            4,
            65_539,
            257,
            129,
            82_004,
            "hot99_empty",
            weight=0.0,
            warmup=0,
            role="correctness",
            component="centroid_update",
            measurement_mode="steady_state",
            visibility="hidden",
        ),
        CaseSpec(
            "hidden_zipf_update",
            "update",
            "online",
            2,
            131_071,
            1025,
            31,
            82_005,
            "zipf",
            weight=0.0,
            warmup=0,
            role="correctness",
            component="centroid_update",
            measurement_mode="steady_state",
            visibility="hidden",
        ),
        CaseSpec(
            "hidden_batched_skew_update",
            "update",
            "online",
            8,
            32_771,
            257,
            65,
            82_006,
            "batched_skew",
            weight=0.0,
            warmup=0,
            role="correctness",
            component="centroid_update",
            measurement_mode="steady_state",
            visibility="hidden",
        ),
        CaseSpec(
            "hidden_streaming_memory_assign",
            "assign",
            "online",
            1,
            256 * 1024,
            4 * 1024,
            128,
            82_007,
            weight=0.0,
            warmup=0,
            role="correctness",
            component="assignment",
            measurement_mode="steady_state",
            verification="sampled_exact_and_streaming_memory",
            visibility="hidden",
        ),
    ]
    return runtime + gates


def hidden_deployment_specs(suite: str) -> list[CaseSpec]:
    if suite == "smoke":
        return [
            CaseSpec(
                "hidden_smoke_deployment_1m_1k_d128",
                "full",
                "online",
                1,
                1 << 20,
                1 << 10,
                128,
                81_050,
                weight=DEPLOYMENT_WEIGHT,
                score_group="deploy_control",
                warmup=0,
                iters=1,
                repeats=1,
                component="deployment_bundle",
                measurement_mode="fresh_process",
                verification="first_run_and_warmed_iteration_sampled_exact",
                visibility="hidden",
            )
        ]
    return [
        benchmark_case_to_spec(case, seed=83_000 + index)
        for index, case in enumerate(HIDDEN_DEPLOYMENT_CASES)
    ]


def locate_root(app_dir: Path) -> Path:
    candidates = [
        app_dir / "workdir" / "flash_kmeans_lab",
        app_dir / "environment" / "workdir" / "flash_kmeans_lab",
        app_dir,
        Path("/app/workdir/flash_kmeans_lab"),
    ]
    for candidate in candidates:
        if (candidate / "kernels").is_dir() and (candidate / "FINAL_VERSION").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot locate flash_kmeans_lab under {app_dir}")


EVALUATION_CONTEXT = Path("/tmp/.sforge-evaluation-context.json")


def load_candidate_version_override(
    path: Path = EVALUATION_CONTEXT,
    *,
    expected_uid: int = 0,
) -> int | None:
    """Return the host-requested version to score instead of FINAL_VERSION.

    Host-side per-version replay scores one archive once per retained
    implementation, which is the only way versions the selector never reached
    ever get a hidden score. The request arrives in SForge's root-only
    evaluation context, so the agent cannot set it, and it is refused outright
    on the interactive rounds the agent sees: an agent-visible result must
    always describe the version its own submission selected.
    """

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        # Contexts predating per-version replay carry no override.
        return None
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("SForge evaluation context is not a regular file")
    if metadata.st_uid != expected_uid:
        raise RuntimeError("SForge evaluation context has an unexpected owner")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise RuntimeError("SForge evaluation context has unexpected permissions")
    if metadata.st_size > 16 * 1024:
        raise RuntimeError("SForge evaluation context is too large")
    context = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(context, dict):
        raise RuntimeError("SForge evaluation context is not an object")
    version = context.get("candidate_version")
    if version is None:
        return None
    if isinstance(version, bool) or not isinstance(version, int) or not 1 <= version <= 999:
        raise RuntimeError("SForge evaluation candidate_version is invalid")
    round_id = context.get("round_id")
    if not isinstance(round_id, str) or not round_id:
        raise RuntimeError("SForge evaluation context lacks round_id")
    for interactive in ("auto-", "agent-"):
        if round_id.startswith(interactive) and len(round_id) > len(interactive):
            raise RuntimeError("candidate_version override is host-side only")
    return version


def resolve_protected_path(app_dir: Path, raw_path: str) -> Path:
    if raw_path.startswith("/"):
        direct = Path(raw_path)
        if direct.exists():
            return direct
        prefixes = {
            "/app/workdir/flash_kmeans_lab/": "workdir/flash_kmeans_lab",
        }
        for prefix, suffix_root in prefixes.items():
            if raw_path.startswith(prefix):
                suffix = raw_path[len(prefix) :]
                for base in (app_dir, app_dir / "environment"):
                    candidate = base / suffix_root / suffix
                    if candidate.exists():
                        return candidate
        return direct
    for candidate in (app_dir / raw_path, app_dir / "environment" / raw_path):
        if candidate.exists():
            return candidate
    return app_dir / raw_path


def protected_failures(app_dir: Path, verifier_dir: Path) -> list[str]:
    manifest = verifier_dir / "protected_hashes.txt"
    if not manifest.is_file():
        return ["missing protected hash manifest"]
    failures: list[str] = []
    paths_seen: set[str] = set()
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            failures.append("malformed protected hash manifest entry")
            continue
        expected, raw_path = parts
        if raw_path in paths_seen:
            failures.append(f"duplicate protected path: {raw_path}")
            continue
        paths_seen.add(raw_path)
        path = resolve_protected_path(app_dir, raw_path)
        if not path.is_file() or path.is_symlink():
            failures.append(f"missing protected file: {raw_path}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"protected file changed: {raw_path}")
    missing = sorted(REQUIRED_PROTECTED_PATHS - paths_seen)
    if missing:
        failures.append(
            "protected hash manifest is incomplete: " + ", ".join(missing)
        )
    return failures


def prepare_runtime_candidate(source: Path, version: int) -> Path:
    account = pwd.getpwnam("agent")
    runtime_dir = Path(f"/tmp/flash-kmeans-candidate-{os.getpid()}")
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(mode=0o700)
    target = runtime_dir / f"flash_kmeans_v{version:03d}.py"
    target.write_bytes(source.read_bytes())
    target.chmod(0o400)
    os.chown(target, account.pw_uid, account.pw_gid)
    os.chown(runtime_dir, account.pw_uid, account.pw_gid)
    return target


def permanent_drop_to_agent() -> None:
    account = pwd.getpwnam("agent")
    os.environ.update(
        {
            "HOME": account.pw_dir,
            "USER": account.pw_name,
            "LOGNAME": account.pw_name,
            "TMPDIR": "/tmp",
            "TRITON_CACHE_DIR": "/tmp/triton-cache-agent",
        }
    )
    os.setgroups([])
    os.setgid(account.pw_gid)
    os.setuid(account.pw_uid)
    os.umask(0o077)
    if os.geteuid() == 0 or os.getuid() == 0:
        raise RuntimeError("candidate evaluator failed to drop root permanently")


def deployment_pass(
    specs: list[CaseSpec],
    source_file: Path,
    *,
    timeout_sec: float,
    candidate_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    frozen: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if (
            spec.stage != "full"
            or spec.component != "deployment_bundle"
            or spec.measurement_mode != "fresh_process"
        ):
            raise ValueError(
                "deployment pass requires full-stage deployment_bundle "
                "specs in fresh_process mode"
            )
        frozen[spec.name] = run_deployment_case(
            source_file,
            case_name=spec.name,
            stage="full",
            b=spec.b,
            n=spec.n,
            k=spec.k,
            d=spec.d,
            seed=spec.seed,
            timeout_sec=timeout_sec,
            warmed_repeats=3,
            candidate_root=candidate_root,
            category=spec.category,
            distribution=spec.distribution,
            tie=spec.tie,
            weight=spec.weight,
            verification="sampled",
        )
        EMPTY_CACHE()
    return frozen


def compose_deployment_rows(
    specs: list[CaseSpec],
    candidates: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        candidate = candidates[spec.name]
        candidate_ok = (
            candidate.get("status") == "ok"
            and candidate.get("correctness_ok") is True
            and candidate.get("input_integrity_ok") is True
            and candidate.get("output_contract_ok") is True
            and candidate.get("cache_was_empty") is True
        )
        limit = memory_limit_bytes(spec)
        peak = int(candidate.get("peak_extra_bytes") or 0)
        memory_ok = candidate_ok and peak <= limit
        memory_detail = memory_diagnostics(peak, limit)
        candidate_warmed_iteration_ms = float(
            candidate.get("warmed_iteration_ms") or 0.0
        )
        candidate_first_run_ms = float(candidate.get("first_run_ms") or 0.0)
        compile_overhead_ratio = (
            candidate_first_run_ms / candidate_warmed_iteration_ms
            if candidate_first_run_ms > 0.0
            and candidate_warmed_iteration_ms > 0.0
            else 0.0
        )
        latency_ok = (
            math.isfinite(candidate_warmed_iteration_ms)
            and candidate_warmed_iteration_ms > 0.0
            and math.isfinite(candidate_first_run_ms)
            and candidate_first_run_ms > 0.0
        )
        all_ok = candidate_ok and memory_ok and latency_ok
        non_memory_ok = candidate_ok and latency_ok
        row: dict[str, Any] = {
            "name": spec.name,
            "stage": spec.stage,
            "component": "deployment_bundle",
            "measurement_mode": spec.measurement_mode,
            "measurement_phase": "warmed_iteration",
            "shape": [spec.b, spec.n, spec.k, spec.d],
            "seed": spec.seed,
            "weight": spec.weight,
            "score_group": spec.score_group,
            "candidate_ms": candidate_warmed_iteration_ms,
            "case_score_ms": candidate_warmed_iteration_ms,
            "candidate_warmed_iteration_ms": candidate_warmed_iteration_ms,
            "candidate_first_run_ms": candidate_first_run_ms,
            "compile_overhead_ratio": compile_overhead_ratio,
            "correctness_ok": bool(candidate.get("correctness_ok")),
            "input_integrity_ok": bool(candidate.get("input_integrity_ok")),
            "memory_ok": memory_ok,
            **memory_detail,
            "deployment_protocol_ok": candidate_ok,
            "peak_extra_bytes": peak,
            "memory_limit_bytes": limit,
            "candidate_initial_cache_empty": (
                candidate.get("cache_was_empty") is True
            ),
            "non_memory_gates_ok": non_memory_ok,
            "all_gates_ok": all_ok,
        }
        if not all_ok:
            if non_memory_ok and not memory_ok:
                row["error"] = (
                    "candidate exceeded the streaming-memory budget by "
                    f"{memory_detail['memory_overage_bytes']} bytes "
                    f"({100.0 * memory_detail['memory_overage_ratio']:.4f}%)"
                )
            else:
                row["error"] = (
                    "Fresh-process deployment protocol failed: "
                    f"candidate={candidate.get('status')}"
                )
        rows.append(row)
    return rows


COMPONENT_WEIGHTS = dict(SCORE_GROUP_WEIGHTS)


def trusted_latency_score(
    rows: list[dict[str, Any]],
    specs: list[CaseSpec],
    *,
    gate_key: str = "all_gates_ok",
) -> tuple[float, dict[str, float]]:
    """Compute each private group GM first, then the weighted group GM.

    Names, groups, and weights come from immutable root-owned ``specs``.  A
    missing, duplicate, unknown, malformed, or failed performance row fails
    closed; no partial group is ever renormalized.
    """

    zero = {name: 0.0 for name in SCORE_GROUP_WEIGHTS}
    expected = {spec.name: spec for spec in specs if spec.weight > 0.0}
    if not expected or len(expected) != sum(spec.weight > 0.0 for spec in specs):
        return 0.0, zero
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.get("name")
        if not isinstance(name, str) or name in indexed:
            return 0.0, zero
        indexed[name] = row
        try:
            row_weight = float(row.get("weight", 0.0))
        except (TypeError, ValueError):
            return 0.0, zero
        if not math.isfinite(row_weight) or row_weight < 0.0:
            return 0.0, zero
        if row_weight > 0.0 and name not in expected:
            return 0.0, zero
    if any(name not in indexed for name in expected):
        return 0.0, zero

    grouped_specs: dict[str, list[CaseSpec]] = {}
    for spec in expected.values():
        if spec.score_group not in SCORE_GROUP_WEIGHTS:
            return 0.0, zero
        grouped_specs.setdefault(spec.score_group, []).append(spec)
    observed_group_weights = {
        group: sum(spec.weight for spec in selected)
        for group, selected in grouped_specs.items()
    }
    if not math.isclose(
        sum(observed_group_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        return 0.0, zero
    if set(grouped_specs) == set(SCORE_GROUP_WEIGHTS):
        for group, expected_weight in SCORE_GROUP_WEIGHTS.items():
            if not math.isclose(
                observed_group_weights[group],
                expected_weight,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return 0.0, zero

    group_values = dict(zero)
    total_log = 0.0
    for group, selected in grouped_specs.items():
        group_weight = observed_group_weights[group]
        if not math.isfinite(group_weight) or group_weight <= 0.0:
            return 0.0, zero
        group_log = 0.0
        for spec in selected:
            row = indexed[spec.name]
            if row.get(gate_key) is not True:
                return 0.0, zero
            if (
                row.get("stage") != spec.stage
                or row.get("component") != spec.component
                or row.get("measurement_mode") != spec.measurement_mode
                or row.get("score_group") != spec.score_group
            ):
                return 0.0, zero
            try:
                row_weight = float(row["weight"])
                latency_ms = float(row["candidate_ms"])
            except (KeyError, TypeError, ValueError):
                return 0.0, zero
            if not math.isclose(
                row_weight, spec.weight, rel_tol=0.0, abs_tol=1e-12
            ):
                return 0.0, zero
            if not math.isfinite(latency_ms) or latency_ms <= 0.0:
                return 0.0, zero
            group_log += (spec.weight / group_weight) * TRUSTED_LOG(latency_ms)
        group_value = TRUSTED_EXP(group_log)
        group_values[group] = group_value
        total_log += group_weight * TRUSTED_LOG(group_value)
    score = TRUSTED_EXP(total_log)
    if not math.isfinite(score) or score <= 0.0:
        return 0.0, zero
    return score, group_values


def trusted_diagnostic_score(
    rows: list[dict[str, Any]],
    specs: list[CaseSpec],
) -> tuple[float, dict[str, float]]:
    """Compute latency feedback only when every non-memory gate passed."""

    return trusted_latency_score(
        rows, specs, gate_key="non_memory_gates_ok"
    )


def trusted_coarse_deployment_latencies(
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Return exact root-only family GMs; the adapter exposes only bands."""

    result = {family: 0.0 for family in COARSE_DEPLOYMENT_FAMILIES}
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.get("name")
        if not isinstance(name, str) or name in indexed:
            return result
        indexed[name] = row
    for family, members in COARSE_DEPLOYMENT_FAMILIES.items():
        selected = [indexed.get(name) for name in members]
        if any(row is None for row in selected):
            continue
        latencies: list[float] = []
        for row in selected:
            assert row is not None
            try:
                latency = float(row.get("candidate_ms") or 0.0)
            except (TypeError, ValueError):
                latencies = []
                break
            if (
                row.get("all_gates_ok") is not True
                or not math.isfinite(latency)
                or latency <= 0.0
            ):
                latencies = []
                break
            latencies.append(latency)
        if latencies:
            result[family] = TRUSTED_EXP(
                sum(TRUSTED_LOG(value) for value in latencies) / len(latencies)
            )
    return result


def trusted_coarse_deployment_statuses(
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    """Return independent, shape-free evidence status for each private family."""

    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        name = row.get("name")
        if isinstance(name, str):
            by_name.setdefault(name, []).append(row)
    statuses: dict[str, str] = {}
    for family, members in COARSE_DEPLOYMENT_FAMILIES.items():
        selected: list[dict[str, Any]] = []
        for name in members:
            matches = by_name.get(name, [])
            if len(matches) != 1:
                selected = []
                break
            selected.append(matches[0])
        if not selected:
            statuses[family] = "not_reached"
        elif all(row.get("all_gates_ok") is True for row in selected):
            statuses[family] = "passed"
        else:
            statuses[family] = "failed"
    return statuses


def summarize_memory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [
        row
        for row in rows
        if isinstance(row.get("memory_ratio"), (int, float))
        and math.isfinite(float(row["memory_ratio"]))
    ]
    if not measured:
        return {
            "max_memory_ratio": 0.0,
            "memory_headroom_ratio": 0.0,
            "memory_overage_ratio": 0.0,
            "memory_pressure": "unavailable",
            "worst_memory_stage": "unavailable",
        }
    worst = max(measured, key=lambda row: float(row["memory_ratio"]))
    ratio = float(worst["memory_ratio"])
    return {
        "max_memory_ratio": ratio,
        "memory_headroom_ratio": 1.0 - ratio,
        "memory_overage_ratio": max(0.0, ratio - 1.0),
        "memory_pressure": worst.get("memory_pressure", "unavailable"),
        # Stage is safe to expose; private case names, shapes, and seeds are not.
        "worst_memory_stage": str(worst.get("stage", "unavailable")),
    }


def evaluate_case(
    spec: CaseSpec,
    assign_fn: Any,
    update_fn: Any,
    pre_candidate_allocated: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": spec.name,
        "stage": spec.stage,
        "category": spec.category,
        "shape": [spec.b, spec.n, spec.k, spec.d],
        "distribution": spec.distribution,
        "seed": spec.seed,
        "weight": spec.weight,
        "score_group": spec.score_group,
        "role": spec.role,
        "component": spec.component,
        "measurement_mode": spec.measurement_mode,
        "verification": spec.verification,
        "warmup": spec.warmup,
        "iters": spec.iters,
        "repeats": spec.repeats,
        "candidate_ms": 0.0,
        "case_score_ms": 0.0,
        "correctness_ok": False,
        "input_integrity_ok": False,
        "memory_ok": False,
        "non_memory_gates_ok": False,
        "all_gates_ok": False,
        "memory_limit_bytes": memory_limit_bytes(spec),
    }
    data: dict[str, torch.Tensor] | None = None
    guards: dict[str, torch.Tensor] | None = None
    try:
        # Shape-matched setup input is distinct from every measured and checked
        # tensor. Persistent scratch retained by the candidate is charged from
        # the pre-import allocation watermark.
        warm_data = make_inputs(replace(spec, seed=spec.seed + 40_000_000))
        warm_guards = clone_guards(warm_data, spec.stage)
        for _ in range(max(1, spec.warmup)):
            warm_output = run_stage(spec.stage, assign_fn, update_fn, warm_data)
            DEVICE_SYNCHRONIZE()
            del warm_output
        unchanged, message = inputs_unchanged(warm_data, warm_guards)
        if not unchanged:
            row["error"] = message
            return row
        del warm_guards
        del warm_data
        EMPTY_CACHE()
        DEVICE_SYNCHRONIZE()
        resident = max(0, int(MEMORY_ALLOCATED()) - pre_candidate_allocated)

        data = make_inputs(replace(spec, seed=spec.seed + 50_000_000))
        guards = clone_guards(data, spec.stage)
        case_harness_bytes = max(
            0,
            int(MEMORY_ALLOCATED()) - pre_candidate_allocated - resident,
        )
        output, call_peak = peak_output(spec, assign_fn, update_fn, data)
        accounted_peak = resident + call_peak
        unchanged, message = inputs_unchanged(data, guards)
        row["input_integrity_ok"] = unchanged
        if not unchanged:
            row["error"] = message
            return row
        correct, message = check_output(spec, output, data)
        del output
        row["correctness_ok"] = correct
        if not correct:
            row["error"] = message
            return row

        measured = benchmark_stable_latency_ms(
            spec,
            assign_fn,
            update_fn,
            warmup=spec.warmup,
            iters=spec.iters,
            repeats=spec.repeats,
            seed_offset=20_000_000,
            validate_outputs=True,
            track_memory=True,
        )
        candidate_ms = float(measured["latency_ms"])
        timed_peak = int(measured["peak_bytes"])
        EMPTY_CACHE()
        DEVICE_SYNCHRONIZE()
        resident = max(
            resident,
            max(
                0,
                int(MEMORY_ALLOCATED())
                - pre_candidate_allocated
                - case_harness_bytes,
            ),
        )
        accounted_peak = max(accounted_peak, resident + int(timed_peak))
        row["timing_policy"] = measured["policy"]
        row["candidate_trial_medians_ms"] = measured["trial_medians_ms"]
        row["candidate_timing_spread_ratio"] = measured["spread_ratio"]
        row["candidate_timing_adaptive_retry"] = measured["adaptive_retry"]
        row["timing_protocol_ok"] = measured["stable"]
        if measured.get("stable") is not True:
            row["error"] = (
                "candidate timing trials remained unstable: "
                f"{measured['trial_medians_ms']}"
            )
            return row
        row["candidate_ms"] = float(candidate_ms)
        row["case_score_ms"] = float(candidate_ms)

        # Final correctness is on an unseen post-benchmark seed and pointer.
        post_data = make_inputs(replace(spec, seed=spec.seed + 90_000_000))
        post_guards = clone_guards(post_data, spec.stage)
        post_output = run_stage(spec.stage, assign_fn, update_fn, post_data)
        DEVICE_SYNCHRONIZE()
        unchanged, message = inputs_unchanged(post_data, post_guards)
        if not unchanged:
            row["input_integrity_ok"] = False
            row["error"] = message
            return row
        correct, message = check_output(spec, post_output, post_data)
        del post_output
        del post_guards
        del post_data
        # Release all verifier-owned case tensors before measuring residual
        # candidate state. Holding an input globally is therefore charged too.
        del guards
        guards = None
        del data
        data = None
        if not correct:
            row["correctness_ok"] = False
            row["error"] = f"post-benchmark correctness: {message}"
            return row
        EMPTY_CACHE()
        DEVICE_SYNCHRONIZE()
        residual = max(0, int(MEMORY_ALLOCATED()) - pre_candidate_allocated)
        accounted_peak = max(accounted_peak, residual)
        row.update(
            {
                "resident_candidate_bytes": resident,
                "single_call_peak_bytes": call_peak,
                "timed_peak_bytes": int(timed_peak),
                "residual_candidate_bytes": residual,
                "accounted_peak_bytes": accounted_peak,
            }
        )
        row.update(
            memory_diagnostics(
                accounted_peak,
                row["memory_limit_bytes"],
            )
        )
        row["memory_ok"] = accounted_peak <= row["memory_limit_bytes"]
        row["non_memory_gates_ok"] = True
        if not row["memory_ok"]:
            row["error"] = (
                "candidate exceeded the streaming-memory budget by "
                f"{row['memory_overage_bytes']} bytes "
                f"({100.0 * row['memory_overage_ratio']:.4f}%)"
            )
            return row
        row["all_gates_ok"] = True
        return row
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    finally:
        if guards is not None:
            del guards
        if data is not None:
            del data
        EMPTY_CACHE()


def evaluate_gate_case(
    spec: CaseSpec,
    assign_fn: Any,
    update_fn: Any,
    pre_candidate_allocated: int,
) -> dict[str, Any]:
    """Run a zero-weight semantic/memory gate without fake performance timing."""
    row: dict[str, Any] = {
        "name": spec.name,
        "stage": spec.stage,
        "category": spec.category,
        "shape": [spec.b, spec.n, spec.k, spec.d],
        "distribution": spec.distribution,
        "seed": spec.seed,
        "weight": 0.0,
        "score_group": spec.score_group,
        "role": spec.role,
        "component": spec.component,
        "measurement_mode": spec.measurement_mode,
        "verification": spec.verification,
        "candidate_ms": 0.0,
        "case_score_ms": 0.0,
        "correctness_ok": False,
        "input_integrity_ok": False,
        "memory_ok": False,
        "non_memory_gates_ok": False,
        "all_gates_ok": False,
        "memory_limit_bytes": memory_limit_bytes(spec),
    }
    data: dict[str, torch.Tensor] | None = None
    guards: dict[str, torch.Tensor] | None = None
    try:
        data = make_inputs(replace(spec, seed=spec.seed + 50_000_000))
        guards = clone_guards(data, spec.stage)
        output, call_peak = peak_output(spec, assign_fn, update_fn, data)
        unchanged, message = inputs_unchanged(data, guards)
        row["input_integrity_ok"] = unchanged
        if not unchanged:
            row["error"] = message
            return row
        correct, message = check_output(spec, output, data)
        del output
        row["correctness_ok"] = correct
        if not correct:
            row["error"] = message
            return row
        del guards
        guards = None
        del data
        data = None
        EMPTY_CACHE()
        DEVICE_SYNCHRONIZE()
        residual = max(0, int(MEMORY_ALLOCATED()) - pre_candidate_allocated)
        accounted_peak = residual + int(call_peak)
        row.update(
            {
                "single_call_peak_bytes": int(call_peak),
                "residual_candidate_bytes": residual,
                "accounted_peak_bytes": accounted_peak,
            }
        )
        row.update(
            memory_diagnostics(
                accounted_peak,
                row["memory_limit_bytes"],
            )
        )
        row["memory_ok"] = accounted_peak <= row["memory_limit_bytes"]
        row["non_memory_gates_ok"] = True
        if not row["memory_ok"]:
            row["error"] = (
                "candidate exceeded the streaming-memory budget by "
                f"{row['memory_overage_bytes']} bytes "
                f"({100.0 * row['memory_overage_ratio']:.4f}%)"
            )
            return row
        row["all_gates_ok"] = True
        return row
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    finally:
        if guards is not None:
            del guards
        if data is not None:
            del data
        EMPTY_CACHE()


def base_detail() -> dict[str, Any]:
    return {
        "schema_version": 6,
        "task": "flash_kmeans_paper_h100_kernel_lab_v7_scratch",
        "score_profile": SCORE_PROFILE,
        "protected_ok": 0.0,
        "source_ok": 0.0,
        "interface_ok": 0.0,
        "correctness": 0.0,
        "input_integrity_ok": 0.0,
        "memory_ok": 0.0,
        "timing_protocol_ok": 0.0,
        "deployment_protocol_ok": 0.0,
        "all_gates_ok": 0.0,
        "score": 0.0,
        "weighted_latency_ms": 0.0,
        "diagnostic_score": 0.0,
        "diagnostic_weighted_latency_ms": 0.0,
        "runtime_weighted_latency_ms": 0.0,
        "diagnostic_runtime_weighted_latency_ms": 0.0,
        "assign_latency_ms": 0.0,
        "update_latency_ms": 0.0,
        "full_latency_ms": 0.0,
        "diagnostic_assign_latency_ms": 0.0,
        "diagnostic_update_latency_ms": 0.0,
        "diagnostic_full_latency_ms": 0.0,
        "first_run_latency_ms": 0.0,
        "warmed_iteration_latency_ms": 0.0,
        "diagnostic_first_run_latency_ms": 0.0,
        "diagnostic_warmed_iteration_latency_ms": 0.0,
        "coarse_deployment_large_k_latency_ms": 0.0,
        "coarse_deployment_large_n_high_d_latency_ms": 0.0,
        "coarse_deployment_family_statuses": {
            "large_k": "not_reached",
            "large_n_high_d": "not_reached",
        },
        "compile_overhead_ratio": 0.0,
        "diagnostic_compile_overhead_ratio": 0.0,
        "max_memory_ratio": 0.0,
        "memory_headroom_ratio": 0.0,
        "memory_overage_ratio": 0.0,
    }


def fail(files: ResultFiles, detail: dict[str, Any], message: str) -> int:
    detail = {**base_detail(), **detail, "error": message, "score": 0.0}
    files.write(detail, 0.0)
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", type=Path, default=Path(os.environ.get("APP_DIR", "/app")))
    parser.add_argument("--log-dir", type=Path, default=Path(os.environ.get("LOG_DIR", "/logs/verifier")))
    parser.add_argument(
        "--suite",
        choices=("full", "smoke"),
        default=os.environ.get("FLASH_KMEANS_VERIFY_SUITE", "full"),
    )
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("FLASH_KMEANS_VERIFY_WARMUP", "3")))
    parser.add_argument("--iters", type=int, default=int(os.environ.get("FLASH_KMEANS_VERIFY_ITERS", "5")))
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("FLASH_KMEANS_VERIFY_REPEATS", "5")))
    parser.add_argument(
        "--deployment-case-timeout",
        type=float,
        default=float(
            os.environ.get("FLASH_KMEANS_DEPLOYMENT_CASE_TIMEOUT", "180")
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = ResultFiles(args.log_dir.resolve())
    suppress_process_output()
    detail = base_detail()
    app_dir = args.app_dir.resolve()
    verifier_dir = Path(__file__).resolve().parent
    try:
        if not torch.cuda.is_available():
            return fail(files, detail, "CUDA is required")
        root = locate_root(app_dir)
        protected = protected_failures(app_dir, verifier_dir)
        if protected:
            return fail(files, {**detail, "protected_failures": protected}, "protected files changed")
        detail["protected_ok"] = 1.0

        admission = scan_submission(root)
        requested_version = admission.get("selected_version")
        versions = admission.get("candidate_versions", [])
        if isinstance(requested_version, int) and 1 <= requested_version <= 999:
            detail["final_version"] = float(requested_version)
            detail["requested_final_version"] = float(requested_version)
        if isinstance(versions, list):
            detail["num_candidate_versions"] = float(len(versions))
        if admission["status"] != "clean":
            return fail(
                files,
                {**detail, "source_problems": admission["problems"]},
                "source admission failed",
            )
        detail["source_ok"] = 1.0
        version = requested_version
        if not isinstance(version, int):
            return fail(files, detail, "FINAL_VERSION is invalid")
        version_override = load_candidate_version_override()
        if version_override is not None:
            if version_override not in versions:
                return fail(
                    files,
                    {**detail, "candidate_version_override": float(version_override)},
                    "requested candidate_version is not in this submission",
                )
            version = version_override
            detail["candidate_version_override"] = float(version_override)
        detail["final_version"] = float(version)
        detail["evaluated_final_version"] = float(version)
        detail["num_candidate_versions"] = float(len(versions))
        source = root / "kernels" / f"flash_kmeans_v{version:03d}.py"
        runtime_candidate = prepare_runtime_candidate(source, version)

        specs = hidden_runtime_specs(args.suite)
        deployment_specs = hidden_deployment_specs(args.suite)
        EMPTY_CACHE()
        DEVICE_SYNCHRONIZE()
        pre_candidate_allocated = int(MEMORY_ALLOCATED())
        candidate_digest = hashlib.sha256(runtime_candidate.read_bytes()).hexdigest()

        # Start each fresh-process candidate while the parent can still read
        # the root-only deployment harness. Every untrusted deployment child
        # irreversibly drops to agent before importing candidate code.
        candidate_deployment = deployment_pass(
            deployment_specs,
            runtime_candidate,
            timeout_sec=args.deployment_case_timeout,
            candidate_root=root,
        )
        if hashlib.sha256(runtime_candidate.read_bytes()).hexdigest() != candidate_digest:
            return fail(
                files,
                detail,
                "candidate changed during fresh-process deployment evaluation",
            )
        deployment_rows = compose_deployment_rows(
            deployment_specs,
            candidate_deployment,
        )

        # Mandatory irreversible parent boundary for steady-state evaluation.
        # Result descriptors are already open and trusted benchmark code is
        # already imported; candidate source is imported only after this drop.
        permanent_drop_to_agent()
        _module, assign_fn, update_fn = load_candidate(runtime_candidate, version)
        detail["interface_ok"] = 1.0
        runtime_rows: list[dict[str, Any]] = []
        for spec in specs:
            if spec.weight <= 0.0:
                runtime_rows.append(
                    evaluate_gate_case(
                        spec,
                        assign_fn,
                        update_fn,
                        pre_candidate_allocated,
                    )
                )
            else:
                runtime_rows.append(
                    evaluate_case(
                        spec,
                        assign_fn,
                        update_fn,
                        pre_candidate_allocated,
                    )
                )
        rows = runtime_rows + deployment_rows
        detail["cases"] = rows
        correctness = all(bool(row.get("correctness_ok")) for row in rows)
        input_ok = all(bool(row.get("input_integrity_ok")) for row in rows)
        memory_ok = all(bool(row.get("memory_ok")) for row in rows)
        all_ok = all(bool(row.get("all_gates_ok")) for row in rows)
        detail["correctness"] = 1.0 if correctness else 0.0
        detail["input_integrity_ok"] = 1.0 if input_ok else 0.0
        detail["memory_ok"] = 1.0 if memory_ok else 0.0
        runtime_timing_ok = all(
            bool(row.get("timing_protocol_ok", True))
            for row in runtime_rows
        )
        detail["timing_protocol_ok"] = 1.0 if runtime_timing_ok else 0.0
        deployment_ok = all(
            bool(row.get("deployment_protocol_ok")) for row in deployment_rows
        )
        detail["deployment_protocol_ok"] = 1.0 if deployment_ok else 0.0
        detail["all_gates_ok"] = 1.0 if all_ok else 0.0
        detail.update(summarize_memory(rows))
        score_specs = specs + deployment_specs
        score, components = trusted_latency_score(rows, score_specs)
        diagnostic_score, diagnostic_components = trusted_diagnostic_score(
            rows, score_specs
        )
        coarse_deployment = trusted_coarse_deployment_latencies(deployment_rows)
        coarse_statuses = trusted_coarse_deployment_statuses(deployment_rows)
        detail["coarse_deployment_family_statuses"] = coarse_statuses
        diagnostic_ok = diagnostic_score > 0.0
        if all_ok and (not math.isfinite(score) or score <= 0.0):
            raise RuntimeError(
                "candidate-only scoring protocol produced no positive latency"
            )
        if not all_ok:
            score = 0.0
        runtime_log = sum(
            float(row["weight"])
            * TRUSTED_LOG(float(row["candidate_ms"]))
            for row in runtime_rows
            if float(row["weight"]) > 0.0 and row.get("all_gates_ok")
        )
        runtime_latency_ms = (
            TRUSTED_EXP(runtime_log / RUNTIME_WEIGHT) if all_ok else 0.0
        )
        diagnostic_runtime_log = sum(
            float(row["weight"])
            * TRUSTED_LOG(float(row["candidate_ms"]))
            for row in runtime_rows
            if float(row["weight"]) > 0.0
            and row.get("non_memory_gates_ok")
        )
        diagnostic_runtime_latency_ms = (
            TRUSTED_EXP(diagnostic_runtime_log / RUNTIME_WEIGHT)
            if diagnostic_ok
            else 0.0
        )
        diagnostic_first_run_latencies = [
            float(row["candidate_first_run_ms"])
            for row in deployment_rows
            if row.get("non_memory_gates_ok")
            and float(row["candidate_first_run_ms"]) > 0.0
        ]
        diagnostic_first_run_latency_ms = (
            TRUSTED_EXP(
                sum(TRUSTED_LOG(value) for value in diagnostic_first_run_latencies)
                / len(diagnostic_first_run_latencies)
            )
            if diagnostic_ok and diagnostic_first_run_latencies
            else 0.0
        )
        diagnostic_compile_overheads = [
            float(row["compile_overhead_ratio"])
            for row in deployment_rows
            if row.get("non_memory_gates_ok")
            and float(row.get("compile_overhead_ratio") or 0.0) > 0.0
        ]
        diagnostic_compile_overhead_ratio = (
            TRUSTED_EXP(
                sum(TRUSTED_LOG(value) for value in diagnostic_compile_overheads)
                / len(diagnostic_compile_overheads)
            )
            if diagnostic_ok and diagnostic_compile_overheads
            else 0.0
        )
        detail.update(
            {
                "score": score,
                "weighted_latency_ms": score,
                "diagnostic_score": diagnostic_score,
                "diagnostic_weighted_latency_ms": diagnostic_score,
                "runtime_weighted_latency_ms": runtime_latency_ms,
                "diagnostic_runtime_weighted_latency_ms": (
                    diagnostic_runtime_latency_ms
                ),
                "assign_latency_ms": components["assign"] if all_ok else 0.0,
                "update_latency_ms": components["update"] if all_ok else 0.0,
                "full_latency_ms": components["full"] if all_ok else 0.0,
                "diagnostic_assign_latency_ms": diagnostic_components["assign"],
                "diagnostic_update_latency_ms": diagnostic_components["update"],
                "diagnostic_full_latency_ms": diagnostic_components["full"],
                "first_run_latency_ms": (
                    TRUSTED_EXP(
                        sum(
                            TRUSTED_LOG(float(row["candidate_first_run_ms"]))
                            for row in deployment_rows
                            if float(row["candidate_first_run_ms"]) > 0.0
                        )
                        / max(
                            1,
                            sum(
                                1
                                for row in deployment_rows
                                if float(row["candidate_first_run_ms"]) > 0.0
                            ),
                        )
                    )
                    if all_ok
                    else 0.0
                ),
                "diagnostic_first_run_latency_ms": (
                    diagnostic_first_run_latency_ms
                ),
                "warmed_iteration_latency_ms": (
                    TRUSTED_EXP(
                        (
                            SCORE_GROUP_WEIGHTS["deploy_control"]
                            * TRUSTED_LOG(components["deploy_control"])
                            + SCORE_GROUP_WEIGHTS["deploy_large_k"]
                            * TRUSTED_LOG(components["deploy_large_k"])
                            + SCORE_GROUP_WEIGHTS["deploy_large_n_high_d"]
                            * TRUSTED_LOG(components["deploy_large_n_high_d"])
                        )
                        / DEPLOYMENT_WEIGHT
                    )
                    if all_ok and all(
                        components[name] > 0.0
                        for name in (
                            "deploy_control",
                            "deploy_large_k",
                            "deploy_large_n_high_d",
                        )
                    )
                    else 0.0
                ),
                "diagnostic_warmed_iteration_latency_ms": (
                    TRUSTED_EXP(
                        sum(
                            SCORE_GROUP_WEIGHTS[name]
                            * TRUSTED_LOG(diagnostic_components[name])
                            for name in (
                                "deploy_control",
                                "deploy_large_k",
                                "deploy_large_n_high_d",
                            )
                        )
                        / DEPLOYMENT_WEIGHT
                    )
                    if all(
                        diagnostic_components[name] > 0.0
                        for name in (
                            "deploy_control",
                            "deploy_large_k",
                            "deploy_large_n_high_d",
                        )
                    )
                    else 0.0
                ),
                "coarse_deployment_large_k_latency_ms": coarse_deployment[
                    "large_k"
                ],
                "coarse_deployment_large_n_high_d_latency_ms": (
                    coarse_deployment["large_n_high_d"]
                ),
                "coarse_deployment_family_statuses": coarse_statuses,
                "compile_overhead_ratio": (
                    diagnostic_compile_overhead_ratio if all_ok else 0.0
                ),
                "diagnostic_compile_overhead_ratio": (
                    diagnostic_compile_overhead_ratio
                ),
                "score_formula": (
                    "exp(sum_g(group_weight_g * log(group_GM_g)))"
                ),
                "primary_score_name": "weighted_latency_ms",
                "primary_score_value": score,
                "primary_score_unit": "ms",
                "primary_score_direction": "minimize",
                "primary_score_formula": (
                    "exp(sum_g(group_weight_g * log(group_GM_g)))"
                ),
                "score_group_weights": dict(SCORE_GROUP_WEIGHTS),
                "suite": args.suite,
                "manifest_timing_authoritative": True,
                "cli_timing_compatibility": {
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "repeats": args.repeats,
                },
                "deployment_case_timeout": args.deployment_case_timeout,
            }
        )
        files.write(detail, score)
        return 0 if all_ok else 1
    except Exception as exc:
        return fail(files, detail, f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
