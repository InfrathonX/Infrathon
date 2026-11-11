#!/usr/bin/env python3
"""Root controller for private EAGLE-3 correctness and throughput evaluation."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import shutil
import signal
import socket
import threading
import time
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import torch

JUDGE_ROOT = Path("/opt/eagle3_judge")
SPEC_ROOT = Path("/app/workdir/spec_lab")
# The Judge image carries a root-owned copy of the seed engine so the caps and
# oracle limits can be re-measured on whatever GPU the Judge is running on.
SEED_REFERENCE_VERSION = 1
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sampling import hmac_sample
from scoring import cell_throughput, geometric_mean
from source_admission import validate_submission


def _signal_session(session_id: int, requested: signal.Signals) -> None:
    """Signal every process group left in a participant worker session."""
    groups: set[int] = set()
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError:
        entries = ()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        process_id = int(entry.name)
        try:
            if os.getsid(process_id) == session_id:
                groups.add(os.getpgid(process_id))
        except (OSError, ProcessLookupError):
            continue
    for group in sorted(groups, reverse=True):
        try:
            os.killpg(group, requested)
        except (OSError, ProcessLookupError):
            pass


def run_worker(
    command: list[str],
    request: dict[str, Any],
    timeout: float,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[bytes]:
    """Exec a worker in a killable session and reap every descendant."""
    # The controller's interpreter is never shared with candidate code. Its
    # clock covers generate, every queued stream, materialisation and transfer.
    clock = time.perf_counter
    deadline = clock() + timeout
    parent, child = socket.socketpair()
    process = subprocess.Popen(
        [*command, "--control-fd", str(child.fileno())],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=environment, start_new_session=True, pass_fds=(child.fileno(),),
    )
    child.close()
    captured = []
    drain = threading.Thread(target=lambda: captured.append(process.communicate()), daemon=True)
    drain.start()
    channel = parent.makefile("rwb")
    def send(value):
        """Flush one finite protocol frame."""
        parent.settimeout(max(0.001, deadline - clock()))
        channel.write((json.dumps(value, allow_nan=False) + "\n").encode())
        channel.flush()
    def receive():
        """Receive one bounded protocol frame."""
        parent.settimeout(max(0.001, deadline - clock()))
        raw = channel.readline(16 * 1024 * 1024 + 1)
        if not raw or len(raw) > 16 * 1024 * 1024 or not raw.endswith(b"\n"):
            drain.join(1)
            diagnostic = captured[0][1].decode(errors="replace")[-2000:] if captured else ""
            raise ValueError("missing or oversized worker protocol frame: " + diagnostic)
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("worker frame must be an object")
        return value
    result = {"schema_version": 1, "cells": {}}
    try:
        # No prompt, manifest, or pending request enters the worker at build.
        send({"policy": request["policy"]})
        built = receive()
        if built.get("kind") != "built":
            raise ValueError("worker did not finish build")
        result["build_s"] = built["build_s"]
        for cell in request["cells"]:
            name, batch_size = cell["cell"], int(cell["batch_size"])
            record = {"measurements": [], "outputs": [], "prompt_lengths": []}
            rows = cell["rows"]
            frames = [("warmup", cell["warmup"])]
            frames += [("measured", rows[i:i + batch_size]) for i in range(0, len(rows), batch_size)]
            frames += [("rerun", list(reversed(rows[:2])))]
            for kind, batch in frames:
                send({"kind": kind, "cell": name, "rows": batch,
                      "max_new_tokens": cell["max_new_tokens"],
                      "enable_thinking": cell["enable_thinking"]})
                if receive().get("kind") != "ready":
                    raise ValueError("worker not ready for generation")
                start = clock()
                send({"kind": "go"})
                response = receive()
                wall = clock() - start
                if response.get("kind") != "result":
                    raise ValueError("missing generation result")
                if kind == "measured":
                    measurement = response["measurement"]
                    measurement["worker_wall_s"] = measurement["wall_s"]
                    measurement["wall_s"] = wall
                    record["measurements"].append(measurement)
                    record["outputs"].extend(response["outputs"])
                    record["prompt_lengths"].extend(response["prompt_lengths"])
                elif kind == "rerun":
                    record["rerun_outputs"] = response["outputs"]
                    record["rerun_expected"] = list(reversed(record["outputs"][:2]))
            result["cells"][name] = record
        send({"kind": "done"})
        drain.join(max(0.001, deadline - clock()))
        if drain.is_alive():
            raise subprocess.TimeoutExpired(command, timeout)
    except (TimeoutError, socket.timeout) as exc:
        raise subprocess.TimeoutExpired(command, timeout) from exc
    finally:
        channel.close()
        parent.close()
        _signal_session(process.pid, signal.SIGKILL)
        drain.join(10)
    stdout, stderr = captured[0] if captured else (b"", b"worker pipe drain failed")
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    completed.worker_result = result
    return completed


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one trusted file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protected(root: Path, spec_root: Path) -> None:
    """Verify every declared Judge/runtime/seed identity before candidate import."""
    hashes = root / "protected_hashes.txt"
    roots = {"judge": root, "worker": spec_root, "seed": root / "seed_reference"}
    for line in hashes.read_text(encoding="utf-8").splitlines():
        expected, label = line.split("  ", 1)
        prefix, relative = label.split(":", 1)
        if prefix not in roots:
            raise RuntimeError(f"protected manifest has an unknown scope: {label}")
        path = roots[prefix] / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"protected file mismatch: {label}")


def seed_reference_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the fixed private warm-up cohort used to measure the seed engine.

    Warm-up rows are committed, disjoint from every scored slice, and identical
    for anchor and final, so the derived caps depend on the device rather than
    on the rotated cohort.
    """
    reference = []
    for cell in cells:
        batch = int(cell["batch_size"])
        rows = list(cell["warmup"])
        usable = (len(rows) // batch) * batch
        if usable < batch:
            raise RuntimeError(f"cell {cell['cell']} has no complete warm-up batch")
        reference.append({**cell, "rows": rows[:usable], "warmup": rows[:batch]})
    return reference


def bootstrap_policy(
    policy: dict[str, Any], cells: list[dict[str, Any]]
) -> dict[str, Any]:
    """Bound the reference pass by the worker timeout, not by a device constant."""
    bound = float(policy["worker_timeout_s"])
    return {
        "build_cap_s": policy["build_cap_s"],
        "wall_caps_s": {cell["cell"]: bound for cell in cells},
    }


def derive_resource_caps(
    policy: dict[str, Any],
    cells: list[dict[str, Any]],
    seed_result: dict[str, Any],
) -> tuple[dict[str, float], dict[str, int], dict[str, Any]]:
    """Turn one on-device seed measurement into same-run relative caps.

    The seed's own generation lengths vary with the prompt, so wall is
    normalized to seconds per generated token and extrapolated to a full
    ``batch_size x max_new_tokens`` batch before the multiple is applied.
    """
    if policy["resource_cap_mode"] != "seed_relative_same_run":
        raise RuntimeError("policy does not declare same-run relative caps")
    wall_multiple = float(policy["wall_cap_seed_multiple"])
    memory_multiple = float(policy["memory_cap_seed_multiple"])
    headroom = int(policy["memory_cap_headroom_bytes"])
    wall_caps: dict[str, float] = {}
    memory_caps: dict[str, int] = {}
    reference: dict[str, Any] = {
        "mode": policy["resource_cap_mode"],
        "wall_cap_seed_multiple": wall_multiple,
        "memory_cap_seed_multiple": memory_multiple,
        "memory_cap_headroom_bytes": headroom,
        "cells": {},
    }
    for cell in cells:
        name = cell["cell"]
        rows = seed_result["cells"][name]["measurements"]
        if not rows:
            raise RuntimeError(f"seed reference produced no measurement for {name}")
        per_token = max(
            float(row["wall_s"]) / max(1, sum(int(v) for v in row["generated_lengths"]))
            for row in rows
        )
        peak = max(int(row["peak_memory_bytes"]) for row in rows)
        full_batch_tokens = int(cell["batch_size"]) * int(cell["max_new_tokens"])
        extrapolated = per_token * full_batch_tokens
        cap = wall_multiple * extrapolated
        if not math.isfinite(cap) or cap <= 0 or peak <= 0:
            raise RuntimeError(f"seed reference for {name} is not a usable measurement")
        wall_caps[name] = cap
        memory_caps[name] = math.ceil(memory_multiple * peak) + headroom
        reference["cells"][name] = {
            "seed_wall_s": [float(row["wall_s"]) for row in rows],
            "seed_generated_tokens": [
                sum(int(v) for v in row["generated_lengths"]) for row in rows
            ],
            "seed_seconds_per_generated_token": per_token,
            "seed_peak_memory_bytes": peak,
            "extrapolated_full_batch_wall_s": extrapolated,
            "wall_cap_s": cap,
            "memory_cap_bytes": memory_caps[name],
        }
    return wall_caps, memory_caps, reference


def seed_relative_oracle_limits(
    policy: dict[str, Any], seed_oracle: dict[str, Any]
) -> tuple[float, list[dict[str, float]]]:
    """Floor the frozen quality limits with the same-run seed disagreement rate.

    The eight tier rates were calibrated as a multiple of the seed's own
    argmax-disagreement rate on one GPU. Keeping them as a floor preserves that
    strictness everywhere and relaxes a tier only when the trusted seed itself
    disagrees more on this device.
    """
    multiple = float(policy["oracle_rate_seed_multiple"])
    p_soft = min(1.0, max(float(policy["p_soft"]), multiple * float(seed_oracle["soft_rate"])))
    seed_rates = {
        float(row["margin"]): float(row["rate"]) for row in seed_oracle["tier_results"]
    }
    tiers = [
        {
            "margin": float(tier["margin"]),
            "max_rate": min(
                1.0,
                max(
                    float(tier["max_rate"]),
                    multiple * seed_rates.get(float(tier["margin"]), 0.0),
                ),
            ),
        }
        for tier in policy.get("tiers") or []
    ]
    return p_soft, tiers


def load_jsonl(path: Path, expected: str) -> list[dict[str, Any]]:
    """Load a commitment-checked private JSONL prompt slice."""
    if sha256(path) != expected:
        raise RuntimeError(f"private prompt commitment mismatch: {path.name}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def first_divergence(left: list[int], right: list[int]) -> int | None:
    """Return the first token index at which two streams differ."""
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def selected_cells(
    private: Path,
    profile: str,
    all_anchor: bool,
    secret: bytes,
) -> list[dict[str, Any]]:
    """Load committed cells and select the hidden anchor cohort."""
    manifest = json.loads((private / f"{profile}_manifest.json").read_text())
    warm_manifest = json.loads((private / "warmup_manifest.json").read_text())
    commitments = json.loads((private / "commitments.json").read_text())
    warm_by_name = {entry["cell"]: entry for entry in warm_manifest["cells"]}
    cells = []
    for entry in manifest["cells"]:
        name = entry["cell"]
        rows = load_jsonl(private / profile / f"{name}.jsonl", commitments[f"{profile}:{name}"])
        if profile == "anchor" and not all_anchor:
            if int(entry["batch_size"]) == 8:
                groups = [
                    rows[index:index + 8]
                    for index in range(0, len(rows), 8)
                ]
                ranked = sorted(
                    groups,
                    key=lambda group: hmac.new(
                        secret,
                        (name + ":" + group[0]["prompt_id"]).encode(),
                        hashlib.sha256,
                    ).digest(),
                )
                rows = ranked[0]
            else:
                rows = hmac_sample(rows, 6, secret, name)
        warm_entry = warm_by_name[name]
        warm_rows = load_jsonl(
            private / "warmup" / f"{name}.jsonl",
            commitments[f"warmup:{name}"],
        )
        cells.append(
            {
                **{
                    key: entry[key]
                    for key in (
                        "cell",
                        "batch_size",
                        "max_new_tokens",
                        "enable_thinking",
                    )
                },
                "rows": rows,
                "warmup": warm_rows,
            }
        )
    return cells


def main(argv: list[str] | None = None) -> int:
    """Run admission, the unprivileged worker, and all trusted gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=SPEC_ROOT)
    parser.add_argument("--candidate-version", type=int)
    parser.add_argument("--profile", choices=("anchor", "final"), default="anchor")
    parser.add_argument("--all-anchor", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-root", type=Path, default=JUDGE_ROOT)
    parser.add_argument("--spec-root", type=Path, default=SPEC_ROOT)
    args = parser.parse_args(argv)
    gate_names = (
        "protected_ok",
        "source_ok",
        "worker_ok",
        "o1_ok",
        "o2_ok",
        "tiers_ok",
        "o3_o4_ok",
        "o5_ok",
        "wall_ok",
        "memory_ok",
        "idle_ok",
    )
    gates = {name: False for name in gate_names}
    report: dict[str, Any] = {
        "schema_version": 1,
        "profile": args.profile,
        "all_anchor": args.all_anchor,
        "gates": gates,
    }
    exit_code = 1
    try:
        verify_protected(args.judge_root, args.spec_root)
        gates["protected_ok"] = True
        admission = validate_submission(args.candidate_root, args.candidate_version)
        report["source_admission"] = admission
        if not admission["ok"]:
            raise RuntimeError("source admission failed")
        gates["source_ok"] = True
        private = args.judge_root / "private_data"
        policy = json.loads((private / "policy.json").read_text())
        vanilla = json.loads((private / "reference_streams.json").read_text()).get("streams", {})
        secret_text = os.environ.get("EAGLE3_COHORT_SECRET", "")
        secret = (
            bytes.fromhex(secret_text)
            if len(secret_text) == 64
            else hashlib.sha256(b"host-smoke-anchor").digest()
        )
        cells = selected_cells(private, args.profile, args.all_anchor, secret)

        # Bind every trusted gate and reference-loader operation after its
        # protected hash passes but before the worker can import candidate code.
        # Actual reference weights remain unloaded until the worker exits.
        sys.path.insert(0, str(args.spec_root))
        from runtime.api import GenerationResult, GenerationSpec, validate_result
        from runtime.loader import (
            build_contract,
            default_head_dir,
            load_target,
            load_tokenizer,
        )
        from runtime.oracle import judge_outputs, teacher_forced_check
        from runtime.prompts import render

        bound_generation_result = GenerationResult
        bound_generation_spec = GenerationSpec
        bound_validate_result = validate_result
        bound_build_contract = build_contract
        bound_default_head_dir = default_head_dir
        bound_load_target = load_target
        bound_load_tokenizer = load_tokenizer
        bound_judge_outputs = judge_outputs
        bound_teacher_forced_check = teacher_forced_check
        bound_render = render

        temp_dir = Path(
            tempfile.mkdtemp(prefix="eagle3-worker-", dir="/tmp")
        )
        temp_dir.chmod(0o777)
        worker_output = temp_dir / "result.json"
        worker = args.judge_root / "verifier/candidate_worker.py"
        timeout = float(policy.get("worker_timeout_s", 4800))
        clean_env = {
            "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": "/tmp",
            "PYTHONPATH": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "EAGLE3_TARGET_REPO": os.environ.get("EAGLE3_TARGET_REPO", "/opt/models/qwen3-8b-hf"),
            "EAGLE3_HEAD_DIR": os.environ.get("EAGLE3_HEAD_DIR", "/opt/models/eagle3-head"),
            "EAGLE3_SPEC_ROOT": str(args.spec_root),
        }
        worker_command = [
            sys.executable,
            str(worker),
            "--candidate-root",
            str(args.candidate_root),
            "--candidate-version",
            str(admission["candidate_version"]),
            "--output",
            str(worker_output),
        ]
        # Measure the Judge's own copy of the seed engine on this device, in this
        # run, before the candidate is imported. Every wall and memory cap below
        # is a multiple of that measurement, so no cap is a frozen GPU constant.
        seed_cells = seed_reference_cells(cells)
        seed_completed = run_worker(
            [
                sys.executable,
                str(worker),
                "--candidate-root",
                str(args.judge_root / "seed_reference"),
                "--candidate-version",
                str(SEED_REFERENCE_VERSION),
                "--output",
                str(temp_dir / "seed_reference.json"),
            ],
            {"policy": bootstrap_policy(policy, seed_cells), "cells": seed_cells},
            timeout,
            clean_env,
        )
        if seed_completed.returncode != 0:
            raise RuntimeError(
                "seed reference worker failed with return code "
                f"{seed_completed.returncode}: "
                + seed_completed.stderr.decode(errors="replace")[-2000:]
            )
        seed_result = seed_completed.worker_result
        wall_caps, memory_caps, seed_reference = derive_resource_caps(
            policy, cells, seed_result
        )
        report["seed_reference"] = seed_reference
        request = {
            "policy": {
                **policy,
                "wall_caps_s": wall_caps,
                "memory_caps_bytes": memory_caps,
            },
            "cells": cells,
        }

        completed = run_worker(worker_command, request, timeout, clean_env)
        report["worker_stderr_tail"] = completed.stderr.decode(errors="replace")[-4000:]
        if completed.returncode != 0:
            raise RuntimeError(f"worker failed with return code {completed.returncode}")
        worker_result = completed.worker_result
        gates["worker_ok"] = True

        # Only now load fresh eager reference weights and run bound gate code.
        head_manifest = json.loads(
            (bound_default_head_dir() / "MANIFEST.json").read_text()
        )
        contract = bound_build_contract(
            head_manifest["revision"],
            draft_model_id=head_manifest["repo_id"],
        )
        reference_model = bound_load_target(contract)
        tokenizer = bound_load_tokenizer(contract)

        # Score the same-run seed streams through the same oracle, then use them
        # as the floor for the frozen quality limits. A device whose kernels make
        # the trusted seed disagree more may not fail a candidate for that alone.
        seed_records = []
        for cell in seed_cells:
            measured = seed_result["cells"][cell["cell"]]
            for output, prompt_length in zip(
                measured["outputs"], measured["prompt_lengths"]
            ):
                seed_records.append(
                    bound_teacher_forced_check(
                        reference_model,
                        torch.tensor(
                            output, dtype=torch.int64, device=contract.device
                        ),
                        int(prompt_length),
                        policy["eps_hard"],
                    )
                )
        seed_oracle = bound_judge_outputs(
            seed_records, policy["eps_hard"], policy["p_soft"], policy.get("tiers")
        )
        effective_p_soft, effective_tiers = seed_relative_oracle_limits(
            policy, seed_oracle
        )
        seed_reference["oracle"] = {
            "oracle_rate_seed_multiple": float(policy["oracle_rate_seed_multiple"]),
            "seed_positions": seed_oracle["positions"],
            "seed_soft_divergences": seed_oracle["soft_divergences"],
            "seed_soft_rate": seed_oracle["soft_rate"],
            "seed_max_margin": seed_oracle["max_margin"],
            "seed_tier_rates": [
                {"margin": row["margin"], "rate": row["rate"]}
                for row in seed_oracle["tier_results"]
            ],
            "frozen_p_soft": policy["p_soft"],
            "effective_p_soft": effective_p_soft,
            "effective_tiers": effective_tiers,
        }
        cell_scores, cell_reports = {}, {}
        all_records = []
        for cell in cells:
            name = cell["cell"]
            result = worker_result["cells"][name]
            spec = bound_generation_spec(int(cell["max_new_tokens"]))
            thinking = bool(cell["enable_thinking"])
            prompts = [
                bound_render(row["messages"], tokenizer, thinking).to(
                    contract.device
                )
                for row in cell["rows"]
            ]
            outputs = [
                torch.tensor(
                    ids,
                    dtype=torch.int64,
                    device=contract.device,
                )
                for ids in result["outputs"]
            ]
            lengths = bound_validate_result(
                prompts,
                bound_generation_result(outputs),
                spec,
                contract,
            )
            measured_lengths = [length for measurement in result["measurements"] for length in measurement["generated_lengths"]]
            if measured_lengths != lengths or len(result["measurements"]) != len(prompts) // int(cell["batch_size"]):
                raise ValueError("worker measurement count or token lengths disagree with outputs")
            records = []
            for prompt, output in zip(prompts, outputs):
                record = bound_teacher_forced_check(
                    reference_model,
                    output,
                    prompt.numel(),
                    policy["eps_hard"],
                )
                record.prompt_ids = [prompt]
                record.result = bound_generation_result([output])
                record.spec = spec
                record.contract = contract
                records.append(record)
            oracle = bound_judge_outputs(
                records,
                policy["eps_hard"],
                effective_p_soft,
                effective_tiers,
            )
            all_records.extend(records)
            throughput = cell_throughput(result["measurements"])
            cell_scores[name] = throughput
            wall_ok = all(
                float(row["wall_s"]) <= wall_caps[name]
                for row in result["measurements"]
            )
            memory_ok = all(
                int(row["peak_memory_bytes"]) <= memory_caps[name]
                for row in result["measurements"]
            )
            idle_ok = all(row.get("device_idle") is True for row in result["measurements"])
            o5 = result["rerun_outputs"] == result["rerun_expected"]
            evidence = []
            for row, output, prompt_len in zip(
                cell["rows"],
                result["outputs"],
                result["prompt_lengths"],
            ):
                expected = vanilla.get(row["prompt_id"])
                if isinstance(expected, list):
                    generated = output[prompt_len:]
                    evidence.append(
                        {
                            "prompt_id": row["prompt_id"],
                            "exact": generated == expected,
                            "first_divergence": first_divergence(
                                generated,
                                expected,
                            ),
                        }
                    )
            cell_reports[name] = {
                "tokens_per_s": throughput,
                "generated_tokens": sum(lengths),
                "generated_lengths": lengths,
                "measurements": [
                    {
                        "wall_s": float(row["wall_s"]),
                        "peak_memory_bytes": int(row["peak_memory_bytes"]),
                        "generated_lengths": [
                            int(length) for length in row["generated_lengths"]
                        ],
                        "device_idle": row.get("device_idle") is True,
                        "pending_current_stream_at_return": row.get("pending_current_stream_at_return"),
                        "return_to_sync_s": row.get("return_to_sync_s"),
                        "worker_wall_s": row.get("worker_wall_s"),
                        "trace": row.get("trace", {}),
                    }
                    for row in result["measurements"]
                ],
                "oracle": oracle,
                "o5_passed": o5,
                "wall_passed": wall_ok,
                "memory_passed": memory_ok,
                "idle_passed": idle_ok,
                "vanilla_stream_evidence": evidence,
            }
        aggregate = bound_judge_outputs(
            all_records,
            policy["eps_hard"],
            effective_p_soft,
            effective_tiers,
        )
        gates.update(
            o1_ok=aggregate["o1_passed"],
            o2_ok=aggregate["o2_passed"],
            tiers_ok=aggregate["tiers_passed"],
            o3_o4_ok=all(
                item["oracle"]["o3_o4_passed"]
                for item in cell_reports.values()
            ),
            o5_ok=all(
                item["o5_passed"] for item in cell_reports.values()
            ),
            wall_ok=all(
                item["wall_passed"] for item in cell_reports.values()
            ),
            memory_ok=all(
                item["memory_passed"] for item in cell_reports.values()
            ),
            idle_ok=all(
                item["idle_passed"] for item in cell_reports.values()
            ),
        )
        passed = all(gates.values())
        report.update(
            status="PASS" if passed else "FAIL",
            qualification_state=policy["status"],
            formal_ready=False,
            build_s=worker_result["build_s"],
            cells=cell_reports,
            cell_tokens_per_s=cell_scores,
            score_tokens_per_s=(
                geometric_mean(cell_scores.values()) if passed else 0.0
            ),
            oracle=aggregate,
        )
        exit_code = 0 if passed else 1
    except Exception as exc:
        report.update(
            status="ERROR",
            formal_ready=False,
            score_tokens_per_s=0.0,
            error=f"{type(exc).__name__}: {exc}",
            trusted_traceback=traceback.format_exc()[-8000:],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if "temp_dir" in locals():
        shutil.rmtree(temp_dir, ignore_errors=True)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
