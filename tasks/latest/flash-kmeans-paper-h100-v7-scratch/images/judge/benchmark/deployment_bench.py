#!/usr/bin/env python3
"""Judge-only isolated fresh-process benchmark for deployment latency.

The public/hidden caller owns every case definition.  This module deliberately
contains no benchmark shapes.  One invocation measures exactly one caller-
supplied case in a fresh Python process with fresh Triton and CUDA disk caches.

Security boundary:

* candidates must pass the normal submission source admission;
* candidate stdout/stderr are discarded and never parsed;
* the child writes one nonce-bound JSON record to a parent-created pipe FD;
* a root child permanently drops to ``agent`` before candidate import.

``first_run_ms`` starts immediately before candidate module import and ends
after the first requested stage plus a trusted device-wide CUDA
synchronization. Input construction and post-timing correctness verification
are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import select
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 3
PROTOCOL = "kmeans_kernel_deployment_v3"
MAX_RESULT_BYTES = 256 * 1024
CASE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
STAGES = frozenset({"assign", "update", "full"})
VERIFICATION_MODES = frozenset({"sampled", "full"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_result(
    *,
    case_name: str,
    stage: str,
    b: int,
    n: int,
    k: int,
    d: int,
    seed: int,
    category: str,
    distribution: str,
    tie: bool,
    weight: float,
    verification: str,
    warmed_repeats: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "error",
        "case_name": case_name,
        "stage": stage,
        "shape": [b, n, k, d],
        "seed": seed,
        "category": category,
        "distribution": distribution,
        "tie": bool(tie),
        "weight": float(weight),
        "verification": verification,
        "warmed_repeats": warmed_repeats,
        "source_kind": "candidate",
        "first_run_ms": None,
        "warmed_iteration_ms": None,
        "warmed_iteration_samples_ms": [],
        "correctness_ok": False,
        "input_integrity_ok": False,
        "output_contract_ok": False,
        "memory_observed": False,
        "candidate_output_suppressed": True,
        "result_channel": "nonce_bound_inherited_pipe_fd",
    }


def _validate_request(
    *,
    case_name: str,
    stage: str,
    b: int,
    n: int,
    k: int,
    d: int,
    seed: int,
    category: str,
    distribution: str,
    weight: float,
    verification: str,
    timeout_sec: float,
    warmed_repeats: int,
) -> None:
    if not CASE_NAME_RE.fullmatch(case_name):
        raise ValueError("case_name must match [A-Za-z0-9_.-]{1,128}")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {sorted(STAGES)}")
    for name, value in (("b", b), ("n", n), ("k", k), ("d", d)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not CASE_NAME_RE.fullmatch(category):
        raise ValueError("category must match [A-Za-z0-9_.-]{1,128}")
    if not CASE_NAME_RE.fullmatch(distribution):
        raise ValueError("distribution must match [A-Za-z0-9_.-]{1,128}")
    if not math.isfinite(float(weight)) or float(weight) < 0.0:
        raise ValueError("weight must be finite and non-negative")
    if verification not in VERIFICATION_MODES:
        raise ValueError(
            f"verification must be one of {sorted(VERIFICATION_MODES)}"
        )
    if not math.isfinite(float(timeout_sec)) or not (0.1 <= float(timeout_sec) <= 7200.0):
        raise ValueError("timeout_sec must be finite and in [0.1, 7200]")
    if (
        isinstance(warmed_repeats, bool)
        or not isinstance(warmed_repeats, int)
        or not (1 <= warmed_repeats <= 5)
    ):
        raise ValueError("warmed_repeats must be an integer in [1, 5]")


def _load_source_admission() -> Any:
    path = Path(__file__).resolve().with_name("source_admission.py")
    spec = importlib.util.spec_from_file_location(
        "_kmeans_deployment_source_admission", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load source_admission.py")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _admit_candidate(
    candidate_file: Path,
    candidate_root: Path | None,
) -> tuple[Path, Path, dict[str, Any]]:
    if candidate_file.is_symlink() or not candidate_file.is_file():
        raise ValueError("candidate_file must be a regular non-symlink file")
    runtime_file = candidate_file.resolve()
    root = (
        candidate_root.resolve()
        if candidate_root is not None
        else runtime_file.parent.parent.resolve()
    )
    if candidate_root is not None and (
        candidate_root.is_symlink() or not candidate_root.is_dir()
    ):
        raise ValueError("candidate_root must be a regular non-symlink directory")

    admission_module = _load_source_admission()
    admission = admission_module.scan_submission(root)
    if admission.get("status") != "clean":
        raise ValueError("candidate source admission rejected")
    version = admission.get("selected_version")
    if not isinstance(version, int):
        raise ValueError("source admission did not select an integer version")
    selected = root / "kernels" / f"flash_kmeans_v{version:03d}.py"
    if not selected.is_file() or selected.is_symlink():
        raise ValueError("selected admitted source is not a regular file")
    # A judge may pass an agent-readable runtime copy.  It is admitted only if
    # byte-identical to the selected source.
    if _sha256(selected) != _sha256(runtime_file):
        raise ValueError("candidate_file differs from the admitted selected source")
    return runtime_file, root, admission


def _validate_child_payload(
    payload: Any,
    *,
    nonce: str,
    case_name: str,
    stage: str,
    shape: list[int],
    source_kind: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("child result is not a JSON object")
    required_identity = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "_nonce": nonce,
        "case_name": case_name,
        "stage": stage,
        "shape": shape,
        "source_kind": source_kind,
    }
    for key, expected in required_identity.items():
        if payload.get(key) != expected:
            raise ValueError(f"child result identity mismatch: {key}")
    if payload.get("status") != "ok":
        error_type = str(payload.get("error_type", "child_error"))[:80]
        raise ValueError(f"deployment child failed: {error_type}")
    for key in ("first_run_ms", "warmed_iteration_ms", "peak_extra_bytes"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"child result has invalid {key}")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"child result has non-finite/negative {key}")
    samples = payload.get("warmed_iteration_samples_ms")
    if not isinstance(samples, list) or not samples:
        raise ValueError("child result has no warmed-iteration samples")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in samples
    ):
        raise ValueError("child result has invalid warmed-iteration sample")
    for key in (
        "correctness_ok",
        "input_integrity_ok",
        "output_contract_ok",
        "memory_observed",
        "candidate_output_suppressed",
        "cache_was_empty",
    ):
        if payload.get(key) is not True:
            raise ValueError(f"child hard gate failed: {key}")
    clean = dict(payload)
    clean.pop("_nonce", None)
    return clean


def _read_result_fd(fd: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    os.set_blocking(fd, False)
    deadline = time.monotonic() + 2.0
    while total <= MAX_RESULT_BYTES and time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.1)
        if not ready:
            continue
        chunk = os.read(fd, min(65536, MAX_RESULT_BYTES + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    data = b"".join(chunks)
    if len(data) > MAX_RESULT_BYTES:
        raise ValueError("child result exceeded the size limit")
    return data


def _kill_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError):
        try:
            process.kill()
        except (AttributeError, ProcessLookupError):
            pass
    try:
        process.communicate(timeout=5.0)
    except Exception:
        pass


def _spawn_child(
    candidate_file: Path,
    *,
    request: dict[str, Any],
    timeout_sec: float,
) -> dict[str, Any]:
    nonce = secrets.token_hex(32)
    parent_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="kmeans-deployment-") as temporary:
        temporary_root = Path(temporary)
        triton_cache = temporary_root / "triton-cache"
        cuda_cache = temporary_root / "cuda-cache"
        child_tmp = temporary_root / "tmp"
        for directory in (triton_cache, cuda_cache, child_tmp):
            directory.mkdir(mode=0o700)
        if any(triton_cache.iterdir()):
            raise RuntimeError("fresh TRITON_CACHE_DIR was unexpectedly non-empty")

        read_fd, write_fd = os.pipe()
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_child",
            "--_result-fd",
            str(write_fd),
            "--_nonce",
            nonce,
            "--candidate-file",
            str(candidate_file),
            "--case-name",
            str(request["case_name"]),
            "--stage",
            str(request["stage"]),
            "--b",
            str(request["b"]),
            "--n",
            str(request["n"]),
            "--k",
            str(request["k"]),
            "--d",
            str(request["d"]),
            "--seed",
            str(request["seed"]),
            "--category",
            str(request["category"]),
            "--distribution",
            str(request["distribution"]),
            "--weight",
            str(request["weight"]),
            "--verification",
            str(request["verification"]),
            "--warmed-repeats",
            str(request["warmed_repeats"]),
            "--cache-root",
            str(temporary_root),
        ]
        if request["tie"]:
            command.append("--tie")
        environment = os.environ.copy()
        environment.update(
            {
                "TRITON_CACHE_DIR": str(triton_cache),
                "CUDA_CACHE_PATH": str(cuda_cache),
                "TMPDIR": str(child_tmp),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        process: subprocess.Popen[Any] | None = None
        try:
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    pass_fds=(write_fd,),
                    cwd=str(candidate_file.parent),
                    env=environment,
                    start_new_session=True,
                )
            finally:
                os.close(write_fd)
        except Exception:
            os.close(read_fd)
            raise
        try:
            process.communicate(timeout=float(timeout_sec))
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            os.close(read_fd)
            return {
                "status": "timeout",
                "error_type": "TimeoutExpired",
                "timeout_sec": float(timeout_sec),
                "subprocess_wall_ms": (time.perf_counter() - parent_started) * 1000.0,
            }
        try:
            encoded = _read_result_fd(read_fd)
        finally:
            os.close(read_fd)
        if process.returncode != 0:
            return {
                "status": "error",
                "error_type": "ChildNonzeroExit",
                "child_returncode": int(process.returncode),
                "subprocess_wall_ms": (time.perf_counter() - parent_started) * 1000.0,
            }
        try:
            payload = json.loads(encoded.decode("utf-8"))
            result = _validate_child_payload(
                payload,
                nonce=nonce,
                case_name=str(request["case_name"]),
                stage=str(request["stage"]),
                shape=[
                    int(request["b"]),
                    int(request["n"]),
                    int(request["k"]),
                    int(request["d"]),
                ],
                source_kind="candidate",
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "subprocess_wall_ms": (time.perf_counter() - parent_started) * 1000.0,
            }
        result["subprocess_wall_ms"] = (time.perf_counter() - parent_started) * 1000.0
        return result


def run_deployment_case(
    candidate_file: str | Path,
    *,
    case_name: str,
    stage: str,
    b: int,
    n: int,
    k: int,
    d: int,
    seed: int,
    timeout_sec: float,
    warmed_repeats: int = 3,
    candidate_root: str | Path | None = None,
    category: str = "deployment",
    distribution: str = "uniform",
    tie: bool = False,
    weight: float = 0.0,
    verification: str = "sampled",
) -> dict[str, Any]:
    """Run one admitted candidate deployment case and return a JSON-safe record."""

    base = _base_result(
        case_name=case_name,
        stage=stage,
        b=b,
        n=n,
        k=k,
        d=d,
        seed=seed,
        category=category,
        distribution=distribution,
        tie=tie,
        weight=weight,
        verification=verification,
        warmed_repeats=warmed_repeats,
    )
    try:
        _validate_request(
            case_name=case_name,
            stage=stage,
            b=b,
            n=n,
            k=k,
            d=d,
            seed=seed,
            category=category,
            distribution=distribution,
            weight=weight,
            verification=verification,
            timeout_sec=timeout_sec,
            warmed_repeats=warmed_repeats,
        )
        source = Path(candidate_file)
        if source.is_symlink() or not source.is_file():
            raise ValueError("candidate_file must be a regular non-symlink file")
        source = source.resolve()
        root_path = Path(candidate_root) if candidate_root is not None else None
        source, _root, admission = _admit_candidate(source, root_path)
        request = {
            "case_name": case_name,
            "stage": stage,
            "b": b,
            "n": n,
            "k": k,
            "d": d,
            "seed": seed,
            "category": category,
            "distribution": distribution,
            "tie": bool(tie),
            "weight": float(weight),
            "verification": verification,
            "warmed_repeats": warmed_repeats,
        }
        result = _spawn_child(
            source,
            request=request,
            timeout_sec=timeout_sec,
        )
        result.setdefault("candidate_sha256", _sha256(source))
        result.setdefault(
            "source_admission",
            str(admission.get("status")),
        )
        for key, value in base.items():
            result.setdefault(key, value)
        return result
    except Exception as exc:
        base["error_type"] = type(exc).__name__
        return base


def _make_stage_inputs(
    torch: Any,
    spec: Any,
    make_labels: Any,
    exact_tie_pairs: Any,
) -> dict[str, Any]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(spec.seed)
    x = torch.randn(
        (spec.b, spec.n, spec.d),
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    data: dict[str, Any] = {"x": x.contiguous()}
    if spec.stage in {"assign", "full"}:
        centroids = torch.randn(
            (spec.b, spec.k, spec.d),
            generator=generator,
            device="cuda",
            dtype=torch.float16,
        ).contiguous()
        if spec.tie and spec.k >= 2:
            tie_pairs = exact_tie_pairs(spec.k)
            for lower, upper in tie_pairs:
                centroids[:, upper].copy_(centroids[:, lower])
            tie_points = min(32, spec.n)
            for point_id in range(tie_points):
                lower, _upper = tie_pairs[point_id % len(tie_pairs)]
                x[:, point_id].copy_(centroids[:, lower])
        x_sq = torch.empty(
            (spec.b, spec.n), device="cuda", dtype=torch.float32
        )
        # Avoid a full-shape FP32 x temporary at large-D deployment points.
        point_chunk = max(
            1024,
            min(
                65536,
                64 * 1024 * 1024 // max(4, spec.b * spec.d * 4),
            ),
        )
        for start in range(0, spec.n, point_chunk):
            end = min(start + point_chunk, spec.n)
            tile = x[:, start:end].float()
            x_sq[:, start:end] = tile.square().sum(dim=-1)
        data.update({"centroids": centroids, "x_sq": x_sq})
    else:
        old_centroids = torch.randn(
            (spec.b, spec.k, spec.d),
            generator=generator,
            device="cuda",
            dtype=torch.float16,
        ).contiguous()
        labels = make_labels(spec, generator, "cuda")
        data.update(
            {
                "labels": labels.contiguous(),
                "old_centroids": old_centroids,
            }
        )
    return data


def _guard_indices(
    torch: Any,
    length: int,
    seed: int,
    *,
    include_prefix: bool = False,
    device: str = "cuda",
) -> Any:
    count = min(length, 257)
    if count == length:
        return torch.arange(length, device=device, dtype=torch.int64)
    sequence = torch.arange(count, device=device, dtype=torch.int64)
    step = 2 * (int(seed) % 1_000_003) + 1
    offset = (int(seed) * 1_103_515_245 + 12_345) % length
    selected = (sequence * step + offset) % length
    if include_prefix:
        prefix = torch.arange(
            min(32, length), device=device, dtype=torch.int64
        )
        selected = torch.cat((prefix, selected))
    return torch.unique(selected, sorted=True)


def _capture_input_guards(torch: Any, spec: Any, data: dict[str, Any]) -> dict[str, Any]:
    names = {
        "assign": ("x", "centroids", "x_sq"),
        "update": ("x", "labels", "old_centroids"),
        "full": ("x", "centroids", "x_sq"),
    }[spec.stage]
    guards: dict[str, Any] = {}
    for guard_number, name in enumerate(names):
        tensor = data[name]
        guard_seed = int(spec.seed) + 104_729 * (guard_number + 1)
        if tensor.ndim >= 2 and tensor.shape[1] == spec.n:
            indices = _guard_indices(
                torch,
                spec.n,
                guard_seed,
                include_prefix=bool(spec.tie),
            )
            sample = tensor.index_select(1, indices).clone()
        else:
            flat = tensor.reshape(-1)
            indices = _guard_indices(torch, flat.numel(), guard_seed)
            sample = flat.index_select(0, indices).clone()
        guards[name] = {
            "version": int(tensor._version),
            "indices": indices,
            "sample": sample,
        }
    return guards


def _inputs_unchanged(
    torch: Any,
    spec: Any,
    data: dict[str, Any],
    guards: dict[str, Any],
) -> bool:
    for name, guard in guards.items():
        tensor = data[name]
        if int(tensor._version) != int(guard["version"]):
            return False
        if tensor.ndim >= 2 and tensor.shape[1] == spec.n:
            current = tensor.index_select(1, guard["indices"])
        else:
            current = tensor.reshape(-1).index_select(0, guard["indices"])
        if not bool(torch.equal(current, guard["sample"])):
            return False
    return True


def _extract_output(torch: Any, spec: Any, output: Any) -> tuple[Any, Any, Any]:
    labels = counts = centroids = None
    if spec.stage == "assign":
        labels = output
    elif spec.stage == "update":
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise ValueError("update output must be (counts, centroids)")
        counts, centroids = output
    else:
        if not isinstance(output, (tuple, list)) or len(output) != 3:
            raise ValueError("full output must be (labels, counts, centroids)")
        labels, counts, centroids = output
    if labels is not None:
        if (
            not isinstance(labels, torch.Tensor)
            or tuple(labels.shape) != (spec.b, spec.n)
            or not labels.is_cuda
            or labels.dtype
            not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        ):
            raise ValueError("labels violate the tensor ABI")
        if labels.numel() and (
            int(labels.min().item()) < 0 or int(labels.max().item()) >= spec.k
        ):
            raise ValueError("labels are outside [0, K)")
    if counts is not None:
        if (
            not isinstance(counts, torch.Tensor)
            or tuple(counts.shape) != (spec.b, spec.k)
            or not counts.is_cuda
            or counts.dtype
            not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        ):
            raise ValueError("counts violate the tensor ABI")
    if centroids is not None:
        if (
            not isinstance(centroids, torch.Tensor)
            or tuple(centroids.shape) != (spec.b, spec.k, spec.d)
            or not centroids.is_cuda
            or centroids.dtype != torch.float16
        ):
            raise ValueError("centroids violate the tensor ABI")
    return labels, counts, centroids


def _sampled_assignment_check(
    torch: Any,
    replace: Any,
    check_output: Any,
    spec: Any,
    labels: Any,
    data: dict[str, Any],
) -> bool:
    indices = _guard_indices(
        torch,
        spec.n,
        int(spec.seed) + 99_991,
        include_prefix=bool(spec.tie),
    )
    sampled_x = data["x"].index_select(1, indices)
    sampled_x_sq = data["x_sq"].index_select(1, indices)
    sampled_labels = labels.index_select(1, indices)
    sampled_spec = replace(
        spec,
        stage="assign",
        component="assignment",
        measurement_mode="steady_state",
        n=int(indices.numel()),
    )
    ok, _message = check_output(
        sampled_spec,
        sampled_labels,
        {
            "x": sampled_x,
            "centroids": data["centroids"],
            "x_sq": sampled_x_sq,
        },
    )
    return bool(ok)


def _chunked_update_check(
    torch: Any,
    spec: Any,
    x: Any,
    labels: Any,
    old_centroids: Any,
    counts: Any,
    centroids: Any,
) -> bool:
    counts_ref = torch.zeros(
        (spec.b, spec.k), device="cuda", dtype=torch.int64
    )
    sums = torch.zeros(
        (spec.b, spec.k, spec.d), device="cuda", dtype=torch.float32
    )
    absolute_sum = torch.zeros((), device="cuda", dtype=torch.float32)
    element_count = spec.b * spec.n * spec.d
    chunk = max(
        1024,
        min(65536, 64 * 1024 * 1024 // max(4, spec.d * 4)),
    )
    for batch in range(spec.b):
        for start in range(0, spec.n, chunk):
            end = min(start + chunk, spec.n)
            ids = labels[batch, start:end].to(torch.int64)
            x_float = x[batch, start:end].float()
            counts_ref[batch] += torch.bincount(ids, minlength=spec.k)
            sums[batch].index_add_(0, ids, x_float)
            absolute_sum += x_float.abs().sum()
    if not bool(torch.equal(counts.to(torch.int64), counts_ref)):
        return False
    means = sums / counts_ref.clamp_min(1).float().unsqueeze(-1)
    means = torch.where(
        (counts_ref == 0).unsqueeze(-1),
        old_centroids.float(),
        means,
    )
    mean_abs = (absolute_sum / max(1, element_count)).float().clamp_min(1e-6)
    rounding = (
        64.0
        * torch.finfo(torch.float32).eps
        * counts_ref.float().sqrt().unsqueeze(-1)
        * mean_abs
    )
    tolerance = 0.003 + 0.003 * means.abs() + rounding
    return bool(
        torch.all(
            torch.abs(centroids.float() - means)
            <= tolerance
        )
    )


def _verify_output(
    torch: Any,
    replace: Any,
    check_output: Any,
    spec: Any,
    output: Any,
    data: dict[str, Any],
    verification: str,
) -> tuple[bool, bool]:
    labels, counts, centroids = _extract_output(torch, spec, output)
    if verification == "full":
        ok, _message = check_output(spec, output, data)
        return bool(ok), True
    if spec.stage == "assign":
        return (
            _sampled_assignment_check(
                torch, replace, check_output, spec, labels, data
            ),
            True,
        )
    if spec.stage == "update":
        return (
            _chunked_update_check(
                torch,
                spec,
                data["x"],
                data["labels"],
                data["old_centroids"],
                counts,
                centroids,
            ),
            True,
        )
    labels_ok = _sampled_assignment_check(
        torch, replace, check_output, spec, labels, data
    )
    update_ok = _chunked_update_check(
        torch,
        spec,
        data["x"],
        labels,
        data["centroids"],
        counts,
        centroids,
    )
    return bool(labels_ok and update_ok), True


def _prepare_cache_for_drop(cache_root: Path) -> str:
    if os.geteuid() != 0:
        return "already_unprivileged"
    import pwd

    account = pwd.getpwnam("agent")
    for path in [cache_root, *cache_root.iterdir()]:
        os.chown(path, account.pw_uid, account.pw_gid)
        path.chmod(0o700)
    if hasattr(os, "setresgid"):
        os.setgroups([])
        os.setresgid(account.pw_gid, account.pw_gid, account.pw_gid)
        os.setresuid(account.pw_uid, account.pw_uid, account.pw_uid)
    else:
        os.setgroups([])
        os.setgid(account.pw_gid)
        os.setuid(account.pw_uid)
    os.umask(0o077)
    if os.geteuid() == 0 or os.getuid() == 0:
        raise RuntimeError("failed to drop root before candidate import")
    return "dropped_to_agent"


def _load_runtime_module(path: Path) -> tuple[Any, Any, Any]:
    module_spec = importlib.util.spec_from_file_location(
        "_kmeans_deployment_source", path
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError("could not construct source module spec")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assign_fn = getattr(module, "assign", None)
    update_fn = getattr(module, "update", None)
    if not callable(assign_fn) or not callable(update_fn):
        raise TypeError("source must export callable assign and update")
    return module, assign_fn, update_fn


def _arm_parent_death_signal() -> None:
    """Make an isolated GPU child die if its verifier parent disappears."""

    import ctypes

    pr_set_pdeathsig = 1
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(pr_set_pdeathsig, signal.SIGKILL, 0, 0, 0) != 0:
        errno_value = ctypes.get_errno()
        raise OSError(errno_value, "prctl(PR_SET_PDEATHSIG) failed")
    # Close the fork/exec race: the parent may have died before prctl ran.
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)


def _child_main(args: argparse.Namespace) -> int:
    _arm_parent_death_signal()
    # Import all trusted runtime dependencies before candidate import.
    from dataclasses import replace

    import torch

    benchmark_dir = str(Path(__file__).resolve().parent)
    if benchmark_dir not in sys.path:
        sys.path.insert(0, benchmark_dir)
    from kernel_bench import (
        CaseSpec,
        DEVICE_SYNCHRONIZE,
        MAX_MEMORY_ALLOCATED,
        MEMORY_ALLOCATED,
        RESET_PEAK_MEMORY,
        check_output,
        exact_tie_pairs,
        make_labels,
        run_stage,
    )

    json_dumps = json.dumps
    os_write = os.write
    wall_clock = time.perf_counter
    median = statistics.median
    result = _base_result(
        case_name=args.case_name,
        stage=args.stage,
        b=args.b,
        n=args.n,
        k=args.k,
        d=args.d,
        seed=args.seed,
        category=args.category,
        distribution=args.distribution,
        tie=args.tie,
        weight=args.weight,
        verification=args.verification,
        warmed_repeats=args.warmed_repeats,
    )
    result["_nonce"] = args._nonce
    result["cache_was_empty"] = False
    try:
        cache_root = Path(args.cache_root).resolve()
        triton_cache = Path(os.environ["TRITON_CACHE_DIR"]).resolve()
        if triton_cache.parent != cache_root or not triton_cache.is_dir():
            raise RuntimeError("TRITON_CACHE_DIR escaped the isolated cache root")
        if any(triton_cache.iterdir()):
            raise RuntimeError("TRITON_CACHE_DIR was not empty at child start")
        result["cache_was_empty"] = True
        spec = CaseSpec(
            args.case_name,
            args.stage,
            args.category,
            args.b,
            args.n,
            args.k,
            args.d,
            args.seed,
            args.distribution,
            args.tie,
            args.weight,
            component="deployment_bundle",
            measurement_mode="fresh_process",
        )
        with torch.no_grad():
            process_initial_allocated = int(MEMORY_ALLOCATED())
            data = _make_stage_inputs(
                torch,
                spec,
                make_labels,
                exact_tie_pairs,
            )
            guards = _capture_input_guards(torch, spec, data)
            DEVICE_SYNCHRONIZE()
            allocated_before = int(MEMORY_ALLOCATED())
            RESET_PEAK_MEMORY()

            privilege_mode = _prepare_cache_for_drop(cache_root)

            first_run_started = wall_clock()
            _module, assign_fn, update_fn = _load_runtime_module(
                Path(args.candidate_file)
            )
            output = run_stage(spec.stage, assign_fn, update_fn, data)
            DEVICE_SYNCHRONIZE()
            first_run_ms = (wall_clock() - first_run_started) * 1000.0
            peak_extra_bytes = max(
                0, int(MAX_MEMORY_ALLOCATED()) - allocated_before
            )

            correctness_ok, output_contract_ok = _verify_output(
                torch,
                replace,
                check_output,
                spec,
                output,
                data,
                args.verification,
            )
            input_integrity_ok = _inputs_unchanged(
                torch, spec, data, guards
            )
            if not correctness_ok:
                raise RuntimeError("post-timing correctness verification failed")
            if not input_integrity_ok:
                raise RuntimeError("post-timing input integrity verification failed")
            del output
            del guards
            del data
            torch.cuda.empty_cache()
            peak_extra_bytes = max(
                peak_extra_bytes,
                max(
                    0,
                    int(MEMORY_ALLOCATED()) - process_initial_allocated,
                ),
            )

            warmed_iteration_samples: list[float] = []
            warmed_peak_extra_bytes = 0
            for index in range(args.warmed_repeats):
                resident_before_input = max(
                    0,
                    int(MEMORY_ALLOCATED()) - process_initial_allocated,
                )
                warmed_spec = replace(
                    spec,
                    seed=spec.seed + 1_000_000 + index,
                )
                warmed_data = _make_stage_inputs(
                    torch,
                    warmed_spec,
                    make_labels,
                    exact_tie_pairs,
                )
                guard = _capture_input_guards(
                    torch, warmed_spec, warmed_data
                )
                DEVICE_SYNCHRONIZE()
                warmed_allocated_before = int(MEMORY_ALLOCATED())
                RESET_PEAK_MEMORY()
                started = wall_clock()
                warmed_output = run_stage(
                    warmed_spec.stage, assign_fn, update_fn, warmed_data
                )
                DEVICE_SYNCHRONIZE()
                elapsed = (wall_clock() - started) * 1000.0
                warmed_peak_extra_bytes = max(
                    warmed_peak_extra_bytes,
                    resident_before_input
                    + max(
                        0,
                        int(MAX_MEMORY_ALLOCATED())
                        - warmed_allocated_before,
                    ),
                )
                warmed_correct, warmed_contract = _verify_output(
                    torch,
                    replace,
                    check_output,
                    warmed_spec,
                    warmed_output,
                    warmed_data,
                    args.verification,
                )
                if not warmed_correct or not warmed_contract:
                    raise RuntimeError(
                        "warmed-iteration correctness verification failed"
                    )
                if not _inputs_unchanged(
                    torch, warmed_spec, warmed_data, guard
                ):
                    raise RuntimeError(
                        "warmed-iteration input integrity verification failed"
                    )
                warmed_iteration_samples.append(elapsed)
                del warmed_output
                del guard
                del warmed_data
                torch.cuda.empty_cache()
                warmed_peak_extra_bytes = max(
                    warmed_peak_extra_bytes,
                    max(
                        0,
                        int(MEMORY_ALLOCATED())
                        - process_initial_allocated,
                    ),
                )
            peak_extra_bytes = max(
                peak_extra_bytes,
                warmed_peak_extra_bytes,
            )
            warmed_iteration_ms = float(median(warmed_iteration_samples))
            if (
                not math.isfinite(first_run_ms)
                or first_run_ms <= 0.0
                or not math.isfinite(warmed_iteration_ms)
                or warmed_iteration_ms <= 0.0
            ):
                raise RuntimeError("timing produced a non-positive/non-finite value")
            result.update(
                {
                    "status": "ok",
                    "first_run_ms": float(first_run_ms),
                    "warmed_iteration_ms": warmed_iteration_ms,
                    "warmed_iteration_samples_ms": [
                        float(value)
                        for value in warmed_iteration_samples
                    ],
                    "correctness_ok": True,
                    "input_integrity_ok": True,
                    "output_contract_ok": bool(output_contract_ok),
                    "memory_observed": True,
                    "peak_extra_bytes": int(peak_extra_bytes),
                    "privilege_mode": privilege_mode,
                    "cache_files_after": sum(
                        1 for path in triton_cache.rglob("*") if path.is_file()
                    ),
                    "first_run_definition": (
                        "candidate module import + first stage call + "
                        "device-wide CUDA synchronize"
                    ),
                    "warmed_iteration_definition": (
                        "median fresh-pointer stage call + device-wide CUDA "
                        "synchronize; input generation excluded"
                    ),
                    "correctness_protocol": (
                        "first run and every warmed iteration: full output "
                        "ABI/range; exact deterministic assignment point "
                        "sample; full chunked centroid reduction"
                        if args.verification == "sampled"
                        else (
                            "kernel_bench.check_output over the complete case "
                            "for the first run and every warmed iteration"
                        )
                    ),
                    "input_integrity_protocol": (
                        "tensor version counters plus deterministic "
                        "boundary/strided samples"
                    ),
                }
            )
    except BaseException as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__[:80]
    encoded = json_dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_RESULT_BYTES:
        encoded = json_dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "status": "error",
                "error_type": "ResultTooLarge",
                "_nonce": args._nonce,
                "case_name": args.case_name,
                "stage": args.stage,
                "shape": [args.b, args.n, args.k, args.d],
                "source_kind": "candidate",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    offset = 0
    while offset < len(encoded):
        offset += os_write(args._result_fd, encoded[offset:])
    os.close(args._result_fd)
    return 0 if result.get("status") == "ok" else 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one isolated K-means deployment benchmark case"
    )
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--b", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--warmed-repeats", type=int, default=3)
    parser.add_argument("--category", default="deployment")
    parser.add_argument("--distribution", default="uniform")
    parser.add_argument("--tie", action="store_true")
    parser.add_argument("--weight", type=float, default=0.0)
    parser.add_argument(
        "--verification", choices=sorted(VERIFICATION_MODES), default="sampled"
    )
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_result-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_nonce", help=argparse.SUPPRESS)
    parser.add_argument("--cache-root", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args._child:
        if args._result_fd is None or not args._nonce or args.cache_root is None:
            return 2
        return _child_main(args)
    result = run_deployment_case(
        args.candidate_file,
        candidate_root=args.candidate_root,
        case_name=args.case_name,
        stage=args.stage,
        b=args.b,
        n=args.n,
        k=args.k,
        d=args.d,
        seed=args.seed,
        timeout_sec=args.timeout_sec,
        warmed_repeats=args.warmed_repeats,
        category=args.category,
        distribution=args.distribution,
        tie=args.tie,
        weight=args.weight,
        verification=args.verification,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result.get("status") == "ok":
        return 0
    if result.get("status") == "timeout":
        return 124
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
