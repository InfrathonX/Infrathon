#!/usr/bin/env python3
"""Run one fresh-process HF/Liger calibration pair through the task driver.

This is host-author tooling.  It is deliberately outside both image build
contexts and never enters the participant or Judge filesystem.  Each role is
launched as a separate two-rank torchrun process; the model, data, FSDP,
optimizer, step, timer, token counter, and CUDA peak probes remain owned by
``runtime.distributed_probe``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import transformers


HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE.parent
REPOSITORY_ROOT = TASK_ROOT.parents[2]
TASK_LAB = TASK_ROOT / "images/work/workdir/orpo_lab"
LIGER_LAB = HERE / "liger_oracle"
FULL_LIGER_ENTRY = HERE / "full_liger_qwen3/run_distributed_probe.py"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
LIGER_COMMIT = "780e76b8115bfd7b78592569853b245a33cc7bf0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _common_driver_arguments(args: argparse.Namespace, output: Path) -> list[str]:
    values = [
        "--model-path",
        str(args.model_path),
        "--data-path",
        str(args.data_path),
        "--expected-data-partition",
        args.partition,
        "--expected-manifest-sha256",
        args.manifest_sha256,
        "--expected-tensors-sha256",
        args.tensors_sha256,
        "--output",
        str(output),
        "--seed",
        str(args.seed),
        "--pair-batch",
        str(args.pair_batch),
        "--warmup-steps",
        str(args.warmup_steps),
        "--timed-steps",
        str(args.timed_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--beta",
        str(args.beta),
    ]
    return values


def _command(role: str, args: argparse.Namespace, output: Path) -> list[str]:
    launcher = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--max-restarts=0",
    ]
    common = _common_driver_arguments(args, output)
    if role == "reference":
        return [
            *launcher,
            "--module",
            "runtime.distributed_probe",
            "--kind",
            "reference",
            "--lab-root",
            str(TASK_LAB),
            *common,
        ]
    if role == "objective_liger":
        return [
            *launcher,
            "--module",
            "runtime.distributed_probe",
            "--kind",
            "candidate",
            "--lab-root",
            str(LIGER_LAB),
            "--candidate-version",
            "1",
            *common,
        ]
    if role == "full_liger":
        return [*launcher, str(FULL_LIGER_ENTRY), *common]
    raise ValueError(f"unsupported role: {role}")


def _launch_plan(treatment: str, order: str) -> tuple[tuple[str, str], ...]:
    """Return ``(worker role, comparator label)`` in process-launch order."""

    comparator_labels = tuple(order.split("-"))
    roles = (
        ("reference", "reference")
        if treatment == "hf_self"
        else tuple(
            "reference" if label == "reference" else treatment
            for label in comparator_labels
        )
    )
    return tuple(zip(roles, comparator_labels, strict=True))


def _launch(
    role: str,
    args: argparse.Namespace,
    output: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    command = _command(role, args, output)
    environment = os.environ.copy()
    python_paths = [
        str(TASK_LAB),
        str(REPOSITORY_ROOT / "vendor/Liger-Kernel/src"),
    ]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_BLOCKING_WAIT": "1",
        }
    )
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=TASK_ROOT,
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
            stdout, stderr = process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        raise RuntimeError(f"{role} exceeded the calibration timeout")
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    launch = {
        "role": role,
        "returncode": process.returncode,
        "controller_wall_seconds": time.monotonic() - started,
        "stdout_sha256": _sha256(stdout_path),
        "stderr_sha256": _sha256(stderr_path),
    }
    if process.returncode != 0:
        raise RuntimeError(f"{role} exited with code {process.returncode}")
    if not output.is_file():
        raise RuntimeError(f"{role} did not create its telemetry file")
    launch["telemetry_sha256"] = _sha256(output)
    return launch


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"correctness telemetry has invalid {field}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"correctness telemetry has non-finite {field}")
    return number


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"correctness telemetry has invalid {field}")
    return value


def _validate_correctness_telemetry(
    payload: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Fail closed on missing/non-finite E2E correctness evidence.

    Numeric equivalence tolerances are intentionally optional in this
    non-formal calibration tool.  Until a fresh HF/HF and HF/Liger cohort
    freezes defensible thresholds, this validates all structural and optimizer
    invariants and reports value deltas rather than pretending a guessed
    tolerance is formal.
    """

    telemetry = payload.get("correctness_telemetry")
    if not isinstance(telemetry, dict) or telemetry.get("schema_version") != 1:
        raise RuntimeError("worker omitted correctness telemetry schema v1")
    if telemetry.get("capture_point") != "warmup_after_backward_before_optimizer_step":
        raise RuntimeError("worker correctness telemetry has an unexpected capture point")
    expected_capture_index = args.warmup_steps - 2 if args.warmup_steps >= 2 else 0
    expected_memory_qualification = (
        "settled_after_final_plain_warmup"
        if args.warmup_steps >= 2
        else "diagnostic_capture_on_only_warmup"
    )
    if telemetry.get("capture_warmup_step_index") != expected_capture_index:
        raise RuntimeError("worker correctness telemetry was captured on an unexpected warm-up")
    if telemetry.get("capture_expected_optimizer_step") != expected_capture_index + 1:
        raise RuntimeError("worker correctness telemetry has an invalid expected optimizer step")
    if telemetry.get("memory_qualification") != expected_memory_qualification:
        raise RuntimeError("worker correctness telemetry has an invalid memory qualification")
    if telemetry.get("post_capture_allocator_settled") is not (args.warmup_steps >= 2):
        raise RuntimeError("worker correctness telemetry has invalid allocator-settle state")
    aux = telemetry.get("aux")
    expected_names = (
        "chosen_logps",
        "rejected_logps",
        "chosen_logits_mean",
        "rejected_logits_mean",
        "nll_loss",
        "chosen_rewards",
        "rejected_rewards",
        "logsigmoid_ratio_mean",
        "log_odds_mean",
    )
    if not isinstance(aux, list) or len(aux) != len(expected_names):
        raise RuntimeError("worker correctness telemetry has incomplete auxiliary evidence")
    normalized_aux: list[dict[str, Any]] = []
    for name, entry in zip(expected_names, aux, strict=True):
        if not isinstance(entry, dict) or entry.get("name") != name:
            raise RuntimeError("worker correctness auxiliary names are not canonical")
        shape = entry.get("shape_per_rank")
        if (
            not isinstance(shape, list)
            or any(isinstance(axis, bool) or not isinstance(axis, int) or axis < 0 for axis in shape)
        ):
            raise RuntimeError(f"worker correctness auxiliary {name} has invalid shape")
        if not isinstance(entry.get("dtype_per_rank"), str) or not entry["dtype_per_rank"]:
            raise RuntimeError(f"worker correctness auxiliary {name} has invalid dtype")
        if entry.get("finite") is not True:
            raise RuntimeError(f"worker correctness auxiliary {name} is not finite")
        normalized_aux.append(
            {
                "name": name,
                "shape_per_rank": shape,
                "dtype_per_rank": entry["dtype_per_rank"],
                "global_numel": _positive_integer(entry.get("global_numel"), f"aux.{name}.global_numel"),
                **{
                    metric: _finite_number(entry.get(metric), f"aux.{name}.{metric}")
                    for metric in ("global_sum", "global_l2_sq", "global_min", "global_max")
                },
            }
        )
    gradients = telemetry.get("gradients")
    if not isinstance(gradients, dict) or gradients.get("finite") is not True:
        raise RuntimeError("worker correctness gradient evidence is missing or non-finite")
    if gradients.get("method") != "FSDP.clip_grad_norm_(max_norm=inf,norm_type=2)":
        raise RuntimeError("worker did not use the trusted FSDP global gradient norm")
    normalized_gradients = {
        "global_l2_norm": _finite_number(gradients.get("global_l2_norm"), "gradients.global_l2_norm"),
        **{
            name: _positive_integer(gradients.get(name), f"gradients.{name}")
            for name in (
                "global_trainable_parameter_tensors",
                "global_trainable_parameter_elements",
                "global_gradient_tensors",
                "global_gradient_elements",
            )
        },
    }
    if (
        normalized_gradients["global_l2_norm"] <= 0.0
        or normalized_gradients["global_gradient_tensors"]
        != normalized_gradients["global_trainable_parameter_tensors"]
        or normalized_gradients["global_gradient_elements"]
        != normalized_gradients["global_trainable_parameter_elements"]
    ):
        raise RuntimeError("worker gradient evidence does not cover all trainable shards")
    update = telemetry.get("optimizer_update")
    if not isinstance(update, dict) or update.get("finite") is not True:
        raise RuntimeError("worker correctness optimizer-update evidence is missing")
    expected_step = expected_capture_index + 1
    def parameter_signature(name: str) -> dict[str, Any]:
        signature = update.get(name)
        if not isinstance(signature, dict):
            raise RuntimeError(f"worker optimizer update has no {name} signature")
        return {
            "global_numel": _positive_integer(signature.get("global_numel"), f"{name}.global_numel"),
            "global_sum": _finite_number(signature.get("global_sum"), f"{name}.global_sum"),
            "global_l2_sq": _finite_number(signature.get("global_l2_sq"), f"{name}.global_l2_sq"),
        }

    before_signature = parameter_signature("parameter_before")
    after_signature = parameter_signature("parameter_after")
    if before_signature["global_numel"] != after_signature["global_numel"]:
        raise RuntimeError("worker optimizer update changed parameter topology")
    normalized_update = {
        "expected_step": _positive_integer(update.get("expected_step"), "optimizer.expected_step"),
        **{
            name: _positive_integer(update.get(name), f"optimizer.{name}")
            for name in (
                "global_active_parameter_tensors",
                "global_state_entries",
                "global_entries_at_expected_step",
            )
        },
        "minimum_step": _finite_number(update.get("minimum_step"), "optimizer.minimum_step"),
        "maximum_step": _finite_number(update.get("maximum_step"), "optimizer.maximum_step"),
        "global_exp_avg_l2_norm": _finite_number(
            update.get("global_exp_avg_l2_norm"), "optimizer.global_exp_avg_l2_norm"
        ),
        "parameter_global_sum_delta": _finite_number(
            update.get("parameter_global_sum_delta"), "optimizer.parameter_global_sum_delta"
        ),
        "parameter_global_l2_sq_delta": _finite_number(
            update.get("parameter_global_l2_sq_delta"), "optimizer.parameter_global_l2_sq_delta"
        ),
        "parameter_signature_changed": update.get("parameter_signature_changed"),
        "parameter_before": before_signature,
        "parameter_after": after_signature,
    }
    if (
        telemetry.get("optimizer_class") != "AdamW"
        or update.get("method") != "trusted_AdamW_state_and_compact_parameter_signature"
        or normalized_update["expected_step"] != expected_step
        or normalized_update["minimum_step"] != float(expected_step)
        or normalized_update["maximum_step"] != float(expected_step)
        or normalized_update["global_state_entries"]
        != normalized_update["global_active_parameter_tensors"]
        or normalized_update["global_entries_at_expected_step"]
        != normalized_update["global_active_parameter_tensors"]
        or normalized_update["global_exp_avg_l2_norm"] <= 0.0
        or not isinstance(normalized_update["parameter_signature_changed"], bool)
    ):
        raise RuntimeError("worker optimizer-update evidence is invalid")

    final = telemetry.get("final_optimizer_update")
    if not isinstance(final, dict) or final.get("finite") is not True:
        raise RuntimeError("worker final optimizer-update witness is missing")

    def final_parameter_signature(name: str) -> dict[str, Any]:
        signature = final.get(name)
        if not isinstance(signature, dict):
            raise RuntimeError(f"worker final optimizer witness has no {name} signature")
        return {
            "global_numel": _positive_integer(signature.get("global_numel"), f"final.{name}.global_numel"),
            "global_sum": _finite_number(signature.get("global_sum"), f"final.{name}.global_sum"),
            "global_l2_sq": _finite_number(signature.get("global_l2_sq"), f"final.{name}.global_l2_sq"),
        }

    final_capture_signature = final_parameter_signature("parameter_at_capture_after")
    final_signature = final_parameter_signature("parameter_final")
    expected_final_step = args.warmup_steps + args.timed_steps
    normalized_final = {
        "expected_step": _positive_integer(final.get("expected_step"), "final.expected_step"),
        **{
            name: _positive_integer(final.get(name), f"final.{name}")
            for name in (
                "global_active_parameter_tensors",
                "global_state_entries",
                "global_entries_at_expected_step",
            )
        },
        "minimum_step": _finite_number(final.get("minimum_step"), "final.minimum_step"),
        "maximum_step": _finite_number(final.get("maximum_step"), "final.maximum_step"),
        "global_exp_avg_l2_norm": _finite_number(
            final.get("global_exp_avg_l2_norm"), "final.global_exp_avg_l2_norm"
        ),
        "global_exp_avg_sq_l2_norm": _finite_number(
            final.get("global_exp_avg_sq_l2_norm"), "final.global_exp_avg_sq_l2_norm"
        ),
        "cumulative_parameter_global_sum_delta": _finite_number(
            final.get("cumulative_parameter_global_sum_delta"),
            "final.cumulative_parameter_global_sum_delta",
        ),
        "cumulative_parameter_global_l2_sq_delta": _finite_number(
            final.get("cumulative_parameter_global_l2_sq_delta"),
            "final.cumulative_parameter_global_l2_sq_delta",
        ),
        "cumulative_parameter_signature_changed": final.get(
            "cumulative_parameter_signature_changed"
        ),
        "parameter_at_capture_after": final_capture_signature,
        "parameter_final": final_signature,
    }
    if (
        final.get("capture_point") != "after_timed_window_and_peak_reads"
        or final.get("method")
        != "trusted_AdamW_final_state_and_compact_parameter_signature"
        or normalized_final["expected_step"] != expected_final_step
        or normalized_final["minimum_step"] != float(expected_final_step)
        or normalized_final["maximum_step"] != float(expected_final_step)
        or normalized_final["global_active_parameter_tensors"]
        != normalized_update["global_active_parameter_tensors"]
        or normalized_final["global_state_entries"]
        != normalized_final["global_active_parameter_tensors"]
        or normalized_final["global_entries_at_expected_step"]
        != normalized_final["global_active_parameter_tensors"]
        or normalized_final["global_exp_avg_l2_norm"] <= 0.0
        or normalized_final["global_exp_avg_sq_l2_norm"] <= 0.0
        or final_capture_signature != after_signature
        or final_signature["global_numel"] != after_signature["global_numel"]
        or final_signature["global_l2_sq"] <= 0.0
        or not isinstance(normalized_final["cumulative_parameter_signature_changed"], bool)
        or normalized_final["cumulative_parameter_signature_changed"] is not True
        or (
            normalized_final["cumulative_parameter_global_sum_delta"] == 0.0
            and normalized_final["cumulative_parameter_global_l2_sq_delta"] == 0.0
        )
        or normalized_final["cumulative_parameter_global_sum_delta"]
        != final_signature["global_sum"] - after_signature["global_sum"]
        or normalized_final["cumulative_parameter_global_l2_sq_delta"]
        != final_signature["global_l2_sq"] - after_signature["global_l2_sq"]
    ):
        raise RuntimeError("worker final optimizer-update witness is invalid")
    return {
        "aux": normalized_aux,
        "gradients": normalized_gradients,
        "optimizer_update": normalized_update,
        "final_optimizer_update": normalized_final,
    }


def _validate_payload(payload: dict[str, Any], role: str, args: argparse.Namespace) -> None:
    if payload.get("status") != "PASS" or payload.get("formal") is not False:
        raise RuntimeError(f"{role} did not return non-formal PASS telemetry")
    if payload.get("world_size") != 2 or payload.get("timed_steps") != args.timed_steps:
        raise RuntimeError(f"{role} world size or timed-step count differs")
    if payload.get("model", {}).get("revision") != MODEL_REVISION:
        raise RuntimeError(f"{role} used an unexpected model revision")
    if payload.get("batch", {}).get("partition") != args.partition:
        raise RuntimeError(f"{role} used an unexpected data partition")
    if payload.get("batch", {}).get("pair_batch_per_rank") != args.pair_batch:
        raise RuntimeError(f"{role} used an unexpected pair batch")
    if payload.get("peak_allocated_bytes_max_rank", 0) <= 0:
        raise RuntimeError(f"{role} omitted peak allocated memory")
    if payload.get("peak_reserved_bytes_max_rank", 0) <= 0:
        raise RuntimeError(f"{role} omitted peak reserved memory")
    if payload.get("throughput_scored_tokens_per_second", 0) <= 0:
        raise RuntimeError(f"{role} omitted scored-token throughput")
    _validate_correctness_telemetry(payload, args)
    if role == "full_liger":
        attestation = payload.get("host_oracle_treatment", {})
        if (
            attestation.get("treatment") != "full_liger_qwen3_plus_fused_orpo"
            or attestation.get("liger_commit") != LIGER_COMMIT
        ):
            raise RuntimeError("full-Liger telemetry lacks its pinned treatment attestation")


def _numeric_delta(expected: float, actual: float) -> dict[str, float]:
    absolute = abs(actual - expected)
    return {
        "reference": expected,
        "candidate": actual,
        "absolute_error": absolute,
        "relative_error": absolute / max(abs(expected), abs(actual), 1e-30),
    }


def _compare_correctness(
    reference: dict[str, Any], candidate: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    expected = _validate_correctness_telemetry(reference, args)
    actual = _validate_correctness_telemetry(candidate, args)
    for name in (
        "global_trainable_parameter_tensors",
        "global_trainable_parameter_elements",
        "global_gradient_tensors",
        "global_gradient_elements",
    ):
        if expected["gradients"][name] != actual["gradients"][name]:
            raise RuntimeError("reference/candidate gradient topology differs")
    for phase in ("parameter_before", "parameter_after"):
        if (
            expected["optimizer_update"][phase]["global_numel"]
            != actual["optimizer_update"][phase]["global_numel"]
        ):
            raise RuntimeError("reference/candidate parameter topology differs")
    for phase in ("parameter_at_capture_after", "parameter_final"):
        if (
            expected["final_optimizer_update"][phase]["global_numel"]
            != actual["final_optimizer_update"][phase]["global_numel"]
        ):
            raise RuntimeError("reference/candidate final parameter topology differs")
    for name in (
        "global_active_parameter_tensors",
        "global_state_entries",
        "global_entries_at_expected_step",
        "expected_step",
    ):
        if (
            expected["final_optimizer_update"][name]
            != actual["final_optimizer_update"][name]
        ):
            raise RuntimeError("reference/candidate final optimizer topology differs")
    aux_checks: list[dict[str, Any]] = []
    numeric_checks: list[dict[str, Any]] = []
    for left, right in zip(expected["aux"], actual["aux"], strict=True):
        if (
            left["name"] != right["name"]
            or left["shape_per_rank"] != right["shape_per_rank"]
            or left["global_numel"] != right["global_numel"]
        ):
            raise RuntimeError("reference/candidate auxiliary structure differs")
        metrics = {
            metric: _numeric_delta(left[metric], right[metric])
            for metric in ("global_sum", "global_l2_sq", "global_min", "global_max")
        }
        aux_checks.append({"name": left["name"], "metrics": metrics})
        numeric_checks.extend(
            {"field": f"aux.{left['name']}.{metric}", **delta}
            for metric, delta in metrics.items()
        )
    for name in (
        "global_l2_norm",
        "global_exp_avg_l2_norm",
        "parameter_global_sum_delta",
    ):
        left = (
            expected["gradients"][name]
            if name in expected["gradients"]
            else expected["optimizer_update"][name]
        )
        right = (
            actual["gradients"][name]
            if name in actual["gradients"]
            else actual["optimizer_update"][name]
        )
        numeric_checks.append({"field": name, **_numeric_delta(left, right)})
    for name in (
        "global_exp_avg_l2_norm",
        "global_exp_avg_sq_l2_norm",
        "cumulative_parameter_global_sum_delta",
        "cumulative_parameter_global_l2_sq_delta",
    ):
        numeric_checks.append(
            {
                "field": f"final_optimizer_update.{name}",
                **_numeric_delta(
                    expected["final_optimizer_update"][name],
                    actual["final_optimizer_update"][name],
                ),
            }
        )
    if args.correctness_rtol is not None:
        assert args.correctness_atol is not None
        failures = [
            check
            for check in numeric_checks
            if check["absolute_error"]
            > args.correctness_atol
            + args.correctness_rtol * abs(float(check["reference"]))
        ]
        if failures:
            raise RuntimeError("correctness value delta exceeds explicitly requested calibration tolerance")
    return {
        "mode": "numeric_tolerance" if args.correctness_rtol is not None else "structural_and_finite_only_pending_freeze",
        "numeric_tolerance": (
            None
            if args.correctness_rtol is None
            else {"rtol": args.correctness_rtol, "atol": args.correctness_atol}
        ),
        "aux_checks": aux_checks,
        "numeric_deltas": numeric_checks,
        "optimizer_step_evidence_checked": True,
    }


def _compare(
    reference: dict[str, Any], candidate: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    immutable_fields = (
        "batch",
        "fsdp",
        "optimizer",
        "global_scored_tokens_per_step",
        "global_non_padding_tokens_per_step",
    )
    for name in immutable_fields:
        if reference.get(name) != candidate.get(name):
            raise RuntimeError(f"reference/candidate {name} differs")
    if reference.get("run_id") == candidate.get("run_id"):
        raise RuntimeError("reference and candidate reused one worker run")
    loss_checks: list[dict[str, Any]] = []
    for phase in ("warmup_rank_mean_losses", "timed_rank_mean_losses"):
        expected_values = reference.get(phase, [])
        actual_values = candidate.get(phase, [])
        if len(expected_values) != len(actual_values):
            raise RuntimeError(f"{phase} lengths differ")
        for index, (expected, actual) in enumerate(
            zip(expected_values, actual_values, strict=True)
        ):
            absolute_error = abs(float(actual) - float(expected))
            allowed = args.loss_atol + args.loss_rtol * abs(float(expected))
            if absolute_error > allowed:
                raise RuntimeError(
                    f"candidate {phase}[{index}] loss error exceeds calibration tolerance"
                )
            loss_checks.append(
                {
                    "phase": phase,
                    "index": index,
                    "absolute_error": absolute_error,
                    "allowed": allowed,
                }
            )
    correctness = _compare_correctness(reference, candidate, args)
    reference_tps = float(reference["throughput_scored_tokens_per_second"])
    candidate_tps = float(candidate["throughput_scored_tokens_per_second"])
    return {
        "loss_tolerance": {"rtol": args.loss_rtol, "atol": args.loss_atol},
        "max_loss_absolute_error": max(row["absolute_error"] for row in loss_checks),
        "loss_checks": loss_checks,
        "candidate_over_reference_scored_throughput_ratio": candidate_tps / reference_tps,
        "candidate_over_reference_peak_allocated_ratio": (
            candidate["peak_allocated_bytes_max_rank"]
            / reference["peak_allocated_bytes_max_rank"]
        ),
        "candidate_over_reference_peak_reserved_ratio": (
            candidate["peak_reserved_bytes_max_rank"]
            / reference["peak_reserved_bytes_max_rank"]
        ),
        "correctness": correctness,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--treatment",
        choices=("objective_liger", "full_liger", "hf_self"),
        required=True,
        help="hf_self launches two fresh HF references to characterize worker noise",
    )
    parser.add_argument("--order", choices=("reference-candidate", "candidate-reference"), required=True)
    parser.add_argument("--repeat-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--partition", choices=("public", "private"), required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--tensors-sha256", required=True)
    parser.add_argument("--pair-batch", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss-rtol", type=float, default=5e-3)
    parser.add_argument("--loss-atol", type=float, default=5e-3)
    parser.add_argument(
        "--correctness-rtol",
        type=float,
        help="optional non-formal numeric telemetry gate; omit until calibrated",
    )
    parser.add_argument(
        "--correctness-atol",
        type=float,
        help="required with --correctness-rtol; omit until calibrated",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.pair_batch < 1 or args.warmup_steps < 1 or args.timed_steps < 1:
        parser.error("pair batch and step counts must be positive")
    if args.timeout_seconds < 60:
        parser.error("timeout must be at least 60 seconds")
    if not args.repeat_id.replace("-", "").replace("_", "").isalnum():
        parser.error("repeat ID may contain only letters, digits, hyphens, and underscores")
    if (args.correctness_rtol is None) != (args.correctness_atol is None):
        parser.error("--correctness-rtol and --correctness-atol must be supplied together")
    if (
        args.correctness_rtol is not None
        and (args.correctness_rtol < 0.0 or args.correctness_atol < 0.0)
    ):
        parser.error("correctness tolerances must be non-negative")
    return args


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.treatment}-{args.partition}-p{args.pair_batch}-{args.repeat_id}-{args.order}"
    report_path = args.output_dir / f"{prefix}.paired.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "classification": "HOST_ONLY_ORACLE_CALIBRATION",
        "formal": False,
        "status": "FAIL",
        "treatment": args.treatment,
        "order": args.order,
        "repeat_id": args.repeat_id,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": transformers.__version__,
            "container_image_id": os.environ.get("ORPO_CALIBRATION_IMAGE_ID", "UNSET"),
            "liger_commit": LIGER_COMMIT,
        },
    }
    try:
        outputs: dict[str, Path] = {}
        launches: list[dict[str, Any]] = []
        for ordinal, (role, label) in enumerate(
            _launch_plan(args.treatment, args.order)
        ):
            # hf_self still launches the reference implementation twice, but
            # its comparator labels follow --order so AB/BA reverses the
            # numerator and denominator instead of being metadata-only.
            output = args.output_dir / f"{prefix}.{ordinal}-{label}.telemetry.json"
            stdout_path = args.output_dir / f"{prefix}.{ordinal}-{label}.stdout.log"
            stderr_path = args.output_dir / f"{prefix}.{ordinal}-{label}.stderr.log"
            launches.append(_launch(role, args, output, stdout_path, stderr_path))
            outputs[label] = output
        reference = json.loads(outputs["reference"].read_text(encoding="utf-8"))
        candidate = json.loads(outputs["candidate"].read_text(encoding="utf-8"))
        _validate_payload(reference, "reference", args)
        _validate_payload(
            candidate,
            "reference" if args.treatment == "hf_self" else args.treatment,
            args,
        )
        comparison = _compare(reference, candidate, args)
        if not all(math.isfinite(float(value)) for value in comparison.values() if isinstance(value, float)):
            raise RuntimeError("comparison contains a non-finite metric")
        report.update(
            status="PASS",
            launches=launches,
            comparison=comparison,
            reference_telemetry_sha256=_sha256(outputs["reference"]),
            candidate_telemetry_sha256=_sha256(outputs["candidate"]),
        )
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
