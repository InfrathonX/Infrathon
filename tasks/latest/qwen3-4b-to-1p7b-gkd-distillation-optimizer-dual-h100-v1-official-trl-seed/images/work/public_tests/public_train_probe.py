#!/usr/bin/env python3
"""Launch fresh reference/candidate two-GPU full-training smoke processes.

This is intentionally a small, deterministic qualification probe.  It is not
the hidden workload, does not produce the benchmark score, and does not claim
statistically stable performance results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
DEFAULT_LAB = Path("/app/workdir/gkd_lab")
SOURCE_LAB = HERE.parent / "workdir/gkd_lab"
DEFAULT_STUDENT_MODEL = Path(os.environ.get("GKD_STUDENT_MODEL_PATH", "/models/Qwen3-1.7B"))
DEFAULT_TEACHER_MODEL = Path(os.environ.get("GKD_TEACHER_MODEL_PATH", "/models/Qwen3-4B"))


def _launch(
    *,
    kind: str,
    lab_root: Path,
    student_model_path: Path,
    teacher_model_path: Path,
    result_path: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--max-restarts=0",
        "--module",
        "runtime.distributed_probe",
        "--kind",
        kind,
        "--student-model-path",
        str(student_model_path),
        "--teacher-model-path",
        str(teacher_model_path),
        "--lab-root",
        str(lab_root),
        "--output",
        str(result_path),
        "--seed",
        str(args.seed),
        "--batch",
        str(args.batch),
        "--sequence-length",
        str(args.sequence_length),
        "--prompt-length",
        str(args.prompt_length),
        "--warmup-steps",
        str(args.warmup_steps),
        "--timed-steps",
        str(args.timed_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--beta",
        str(args.beta),
        "--temperature",
        str(args.temperature),
    ]
    if args.candidate_version is not None:
        command.extend(["--candidate-version", str(args.candidate_version)])
    if args.data_path is not None:
        command.extend(
            [
                "--data-path",
                str(args.data_path),
                "--expected-data-partition",
                args.expected_data_partition,
                "--expected-manifest-sha256",
                args.expected_manifest_sha256,
                "--expected-tensors-sha256",
                args.expected_tensors_sha256,
            ]
        )
    environment = os.environ.copy()
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(lab_root) if not old_pythonpath else f"{lab_root}{os.pathsep}{old_pythonpath}"
    )
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    environment["TORCH_NCCL_BLOCKING_WAIT"] = "1"

    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=lab_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        raise RuntimeError(
            f"{kind} torchrun exceeded {args.timeout_seconds}s; stderr tail: {stderr[-4000:]}"
        )
    wall_seconds = time.monotonic() - started
    launch = {
        "kind": kind,
        "launcher_pid": process.pid,
        "returncode": process.returncode,
        "controller_wall_seconds": wall_seconds,
    }
    if process.returncode != 0:
        raise RuntimeError(
            f"{kind} torchrun failed with code {process.returncode}; "
            f"stdout tail: {stdout[-2000:]}; stderr tail: {stderr[-4000:]}"
        )
    if not result_path.is_file():
        raise RuntimeError(f"{kind} torchrun succeeded without telemetry JSON")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return payload, launch


def _validate_run(
    payload: dict[str, Any],
    expected_kind: str,
    timed_steps: int,
    *,
    args: argparse.Namespace,
) -> None:
    if payload.get("status") != "PASS":
        raise RuntimeError(f"{expected_kind} worker did not report PASS")
    if payload.get("formal") is not False:
        raise RuntimeError("public smoke telemetry must explicitly be non-formal")
    if payload.get("classification") != "PUBLIC_FULL_MODEL_SMOKE_DIAGNOSTIC_ONLY":
        raise RuntimeError("unexpected telemetry classification")
    if payload.get("objective_kind") != expected_kind or payload.get("world_size") != 2:
        raise RuntimeError("worker kind/world size differs from controller request")
    candidate_version = payload.get("candidate_version")
    if expected_kind == "reference" and candidate_version is not None:
        raise RuntimeError("trusted reference unexpectedly selected a candidate version")
    if expected_kind == "candidate" and (
        not isinstance(candidate_version, int) or candidate_version <= 0
    ):
        raise RuntimeError("candidate worker did not report a positive selected version")
    fsdp = payload.get("fsdp", {})
    if (
        fsdp.get("api") != "FSDP1"
        or fsdp.get("sharding_strategy") != "FULL_SHARD"
        or fsdp.get("use_orig_params") is not True
        or fsdp.get("parameter_dtype") != "bfloat16"
    ):
        raise RuntimeError("worker did not use the required BF16 FSDP1 configuration")
    model = payload.get("model", {})
    if model.get("tied_lm_head") is not True or model.get("unique_parameters") != 1_720_574_976:
        raise RuntimeError("worker did not validate the pinned trainable tied-head student")
    teacher = payload.get("teacher_model", {})
    if (
        teacher.get("tied_lm_head") is not True
        or teacher.get("unique_parameters") != 4_022_468_096
        or teacher.get("trainable") is not False
        or teacher.get("wrapped_in_fsdp") is not False
        or teacher.get("inside_timed_window") is not True
    ):
        raise RuntimeError("worker did not validate the pinned frozen teacher")
    teacher_witness = payload.get("correctness_telemetry", {}).get("teacher_witness", {})
    if teacher_witness.get("unchanged") is not True or teacher_witness.get(
        "identical_across_ranks"
    ) is not True:
        raise RuntimeError("worker did not witness an unchanged replicated teacher")
    identities = payload.get("rank_identities", [])
    completion = payload.get("rank_completion", [])
    if sorted(item.get("rank") for item in identities) != [0, 1]:
        raise RuntimeError("missing rank identities")
    if sorted(item.get("rank") for item in completion) != [0, 1] or any(
        item.get("status") != "PASS" for item in completion
    ):
        raise RuntimeError("not every rank completed")
    durations = payload.get("rank_max_step_seconds", [])
    scored_tokens = payload.get("global_scored_tokens_per_step", [])
    non_padding_tokens = payload.get("global_non_padding_tokens_per_step", [])
    losses = payload.get("timed_rank_mean_losses", [])
    if not (
        len(durations)
        == len(scored_tokens)
        == len(non_padding_tokens)
        == len(losses)
        == timed_steps
    ):
        raise RuntimeError("timed telemetry length differs from controller-owned step count")
    if any(not isinstance(value, (int, float)) or value <= 0 for value in durations):
        raise RuntimeError("worker emitted an invalid duration")
    if any(not isinstance(value, int) or value <= 0 for value in scored_tokens):
        raise RuntimeError("worker emitted an invalid scored-token count")
    if any(not isinstance(value, int) or value <= 0 for value in non_padding_tokens):
        raise RuntimeError("worker emitted an invalid non-padding-token count")
    if any(scored > non_padding for scored, non_padding in zip(
        scored_tokens, non_padding_tokens, strict=True
    )):
        raise RuntimeError("worker scored-token count exceeds its non-padding count")
    if payload.get("total_global_scored_tokens") != sum(scored_tokens):
        raise RuntimeError("worker scored-token total differs from per-step counts")
    if payload.get("total_global_non_padding_tokens") != sum(non_padding_tokens):
        raise RuntimeError("worker non-padding total differs from per-step counts")
    scored_throughput = payload.get("throughput_scored_tokens_per_second")
    if not isinstance(scored_throughput, (int, float)) or scored_throughput <= 0:
        raise RuntimeError("worker emitted no positive scored-token throughput")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in losses):
        raise RuntimeError("worker emitted a non-finite loss")
    if payload.get("peak_allocated_bytes_max_rank", 0) <= 0:
        raise RuntimeError("worker emitted no allocated-memory telemetry")
    if payload.get("peak_reserved_bytes_max_rank", 0) <= 0:
        raise RuntimeError("worker emitted no reserved-memory telemetry")
    batch = payload.get("batch", {})
    expected_source = (
        "frozen_tldr"
        if args.data_path is not None
        else "deterministic_synthetic_pretokenized_sft"
    )
    if (
        batch.get("source") != expected_source
        or batch.get("row_order_on_disk") != "single_sequence_per_record"
        or batch.get("labels_contract") != "completion_only"
    ):
        raise RuntimeError("worker batch source/order differs from controller request")
    token_contract = batch.get("task_owned_token_counting", {})
    if token_contract.get("primary") != "labels[:, 1:] != -100":
        raise RuntimeError("worker did not use task-owned scored completion tokens")
    if args.data_path is not None:
        if (
            batch.get("partition") != args.expected_data_partition
            or batch.get("manifest_sha256") != args.expected_manifest_sha256.lower()
            or batch.get("tensors_sha256") != args.expected_tensors_sha256.lower()
        ):
            raise RuntimeError("worker frozen-data identity differs from controller request")
        expected_records = (
            (args.warmup_steps + args.timed_steps) * 2 * args.batch
        )
        if batch.get("selected_unique_record_count") != expected_records:
            raise RuntimeError("worker selected an unexpected frozen record count")


def _assert_equivalent_runs(
    reference: dict[str, Any], candidate: dict[str, Any], *, rtol: float, atol: float
) -> dict[str, Any]:
    reference_fingerprint = reference["batch"].get("fingerprint_all_ranks")
    candidate_fingerprint = candidate["batch"].get("fingerprint_all_ranks")
    if (
        not isinstance(reference_fingerprint, str)
        or len(reference_fingerprint) != 64
        or reference_fingerprint != candidate_fingerprint
    ):
        raise RuntimeError("reference and candidate all-rank batch fingerprints differ")
    if reference["batch"] != candidate["batch"]:
        raise RuntimeError("reference and candidate did not consume identical batch specifications")
    if reference["global_non_padding_tokens_per_step"] != candidate[
        "global_non_padding_tokens_per_step"
    ]:
        raise RuntimeError("reference and candidate token accounting differs")
    if reference["global_scored_tokens_per_step"] != candidate[
        "global_scored_tokens_per_step"
    ]:
        raise RuntimeError("reference and candidate scored-token accounting differs")
    if reference["model"]["revision"] != candidate["model"]["revision"]:
        raise RuntimeError("reference and candidate model revisions differ")
    if reference["run_id"] == candidate["run_id"]:
        raise RuntimeError("reference and candidate were not fresh worker runs")
    reference_pids = {item["pid"] for item in reference["rank_identities"]}
    candidate_pids = {item["pid"] for item in candidate["rank_identities"]}
    if reference_pids & candidate_pids:
        raise RuntimeError("reference and candidate reused a rank process")

    compared_losses = []
    for phase in ("warmup_rank_mean_losses", "timed_rank_mean_losses"):
        reference_losses = reference[phase]
        candidate_losses = candidate[phase]
        if len(reference_losses) != len(candidate_losses):
            raise RuntimeError(f"{phase} lengths differ")
        for index, (expected, actual) in enumerate(zip(reference_losses, candidate_losses, strict=True)):
            absolute_error = abs(actual - expected)
            allowed = atol + rtol * abs(expected)
            if absolute_error > allowed:
                raise RuntimeError(
                    f"candidate {phase}[{index}] differs from reference: "
                    f"abs_error={absolute_error:.6g}, allowed={allowed:.6g}"
                )
            compared_losses.append(
                {
                    "phase": phase,
                    "index": index,
                    "reference": expected,
                    "candidate": actual,
                    "absolute_error": absolute_error,
                    "allowed": allowed,
                }
            )
    return {
        "loss_tolerance": {"rtol": rtol, "atol": atol},
        "loss_checks": compared_losses,
    }


def _preflight_frozen_data(
    args: argparse.Namespace, lab_root: Path
) -> dict[str, Any] | None:
    """Validate identity, structure, and capacity before launching torchrun."""

    if args.data_path is None:
        return None
    data_path = args.data_path.expanduser().resolve(strict=True)
    inserted = str(lab_root) not in sys.path
    if inserted:
        sys.path.insert(0, str(lab_root))
    try:
        from runtime.train_step import load_frozen_sft_dataset

        dataset = load_frozen_sft_dataset(
            data_path,
            expected_partition=args.expected_data_partition,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_tensors_sha256=args.expected_tensors_sha256,
            ignore_index=-100,
        )
        dataset.require_capacity(
            total_steps=args.warmup_steps + args.timed_steps,
            batch=args.batch,
            world_size=2,
        )
    finally:
        if inserted:
            sys.path.remove(str(lab_root))
    args.data_path = data_path
    return {
        "source": "frozen_tldr",
        "resolved_path": str(data_path),
        "partition": dataset.partition,
        "manifest_sha256": dataset.manifest_sha256,
        "tensors_sha256": dataset.tensors_sha256,
        "record_count": dataset.record_count,
        "sequence_length": dataset.sequence_length,
        "partition_task_owned_scored_tokens": dataset.scored_tokens,
        "partition_task_owned_non_padding_tokens": dataset.non_padding_tokens,
        "records_required_by_run": (
            (args.warmup_steps + args.timed_steps) * 2 * args.batch
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lab-root",
        type=Path,
        default=DEFAULT_LAB if DEFAULT_LAB.is_dir() else SOURCE_LAB,
    )
    parser.add_argument(
        "--student-model-path",
        "--model-path",
        dest="student_model_path",
        type=Path,
        default=DEFAULT_STUDENT_MODEL,
    )
    parser.add_argument(
        "--teacher-model-path",
        dest="teacher_model_path",
        type=Path,
        default=DEFAULT_TEACHER_MODEL,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="partition directory containing manifest.json and tensors.safetensors",
    )
    parser.add_argument(
        "--expected-data-partition",
        "--expected-partition",
        dest="expected_data_partition",
        choices=("public", "private"),
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        "--expected-data-manifest-sha256",
        dest="expected_manifest_sha256",
    )
    parser.add_argument(
        "--expected-tensors-sha256",
        "--expected-data-sha256",
        "--expected-data-tensors-sha256",
        dest="expected_tensors_sha256",
    )
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--candidate-version", type=int)
    parser.add_argument("--batch", "--pair-batch", dest="batch", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--timed-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--loss-rtol", type=float, default=5e-3)
    parser.add_argument("--loss-atol", type=float, default=5e-3)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--order",
        choices=("reference-candidate", "candidate-reference"),
        default="reference-candidate",
        help="counterbalanced role order for public noise diagnosis",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.timeout_seconds < 60:
        parser.error("timeout must be at least 60 seconds")
    if args.loss_rtol < 0.0 or args.loss_atol < 0.0:
        parser.error("loss tolerances cannot be negative")
    if args.batch < 1:
        parser.error("batch must be positive")
    if not 0.0 <= args.beta <= 1.0:
        parser.error("beta must be in [0, 1]")
    if not args.temperature > 0.0:
        parser.error("temperature must be positive")
    if args.candidate_version is not None and not 1 <= args.candidate_version <= 999:
        parser.error("candidate version must be in [1, 999]")
    expected_data = (
        args.expected_data_partition,
        args.expected_manifest_sha256,
        args.expected_tensors_sha256,
    )
    if args.data_path is not None and any(value is None for value in expected_data):
        parser.error(
            "--data-path requires expected partition, manifest sha256, and tensors sha256"
        )
    if args.data_path is None and any(value is not None for value in expected_data):
        parser.error("expected frozen-data fields require --data-path")
    return args


def main() -> int:
    args = parse_args()
    report: dict[str, Any]
    try:
        lab_root = args.lab_root.expanduser().resolve(strict=True)
        student_model_path = args.student_model_path.expanduser().resolve(strict=True)
        teacher_model_path = args.teacher_model_path.expanduser().resolve(strict=True)
        data_preflight = _preflight_frozen_data(args, lab_root)
        if torch.cuda.device_count() != 2:
            raise RuntimeError(
                "public full-model smoke requires exactly two visible CUDA GPUs; "
                "bind the container/command to the selected GPU pair"
            )
        with tempfile.TemporaryDirectory(prefix="gkd-public-train-") as temporary:
            result_dir = Path(temporary)
            runs: dict[str, dict[str, Any]] = {}
            launches: list[dict[str, Any]] = []
            for kind in args.order.split("-"):
                payload, launch = _launch(
                    kind=kind,
                    lab_root=lab_root,
                    student_model_path=student_model_path,
                    teacher_model_path=teacher_model_path,
                    result_path=result_dir / f"{kind}.json",
                    args=args,
                )
                runs[kind] = payload
                launches.append(launch)
            reference = runs["reference"]
            candidate = runs["candidate"]
        _validate_run(reference, "reference", args.timed_steps, args=args)
        _validate_run(candidate, "candidate", args.timed_steps, args=args)
        equivalence = _assert_equivalent_runs(
            reference,
            candidate,
            rtol=args.loss_rtol,
            atol=args.loss_atol,
        )
        reference_tps = reference["throughput_scored_tokens_per_second"]
        candidate_tps = candidate["throughput_scored_tokens_per_second"]
        report = {
            "schema_version": 1,
            "status": "PASS",
            "classification": "PUBLIC_FULL_MODEL_SMOKE_DIAGNOSTIC_ONLY",
            "formal": False,
            "note": (
                "Frozen TLDR public smoke only; this is not a benchmark score."
                if args.data_path is not None
                else "Synthetic public smoke only; this is not a benchmark score."
            ),
            "primary_throughput_unit": "scored_predicted_completion_tokens_per_second",
            "data_preflight": data_preflight,
            "equivalence": equivalence,
            "diagnostics": {
                "candidate_over_reference_scored_throughput_ratio": (
                    candidate_tps / reference_tps
                ),
                "candidate_over_reference_peak_allocated_ratio": (
                    candidate["peak_allocated_bytes_max_rank"]
                    / reference["peak_allocated_bytes_max_rank"]
                ),
                "candidate_over_reference_peak_reserved_ratio": (
                    candidate["peak_reserved_bytes_max_rank"]
                    / reference["peak_reserved_bytes_max_rank"]
                ),
            },
            "measurement_guidance": {
                "single_pair_absolute_deviation_from_one": {
                    "median": 0.0027,
                    "p90": 0.0140,
                    "max": 0.0253,
                },
                "median_of_four_pairs_absolute_deviation_from_one": {
                    "median": 0.0006,
                    "max": 0.0170,
                },
                "note": (
                    "The formal score is the median of four counterbalanced "
                    "pairs per row, which is why it is far steadier than one "
                    "pair. Treat a single-pair difference under about 1.5% as "
                    "indistinguishable from noise, and re-run before "
                    "concluding anything from it."
                ),
                "gradient_global_l2_norm_fp64": {
                    "reference": reference.get("correctness_telemetry", {})
                    .get("gradients", {})
                    .get("global_l2_norm_fp64"),
                    "candidate": candidate.get("correctness_telemetry", {})
                    .get("gradients", {})
                    .get("global_l2_norm_fp64"),
                },
            },
            "role_order": args.order,
            "launches": launches,
            "reference": reference,
            "candidate": candidate,
        }
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "FAIL",
            "classification": "PUBLIC_FULL_MODEL_SMOKE_DIAGNOSTIC_ONLY",
            "formal": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
