#!/usr/bin/env python3
"""Public correctness qualification and optional single-GPU objective probe."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
DEFAULT_LAB = Path("/app/workdir/gkd_lab")
SOURCE_LAB = HERE.parent / "workdir/gkd_lab"
LAB_ROOT = Path(os.environ.get("GKD_LAB_ROOT", DEFAULT_LAB if DEFAULT_LAB.is_dir() else SOURCE_LAB))
CASES_PATH = HERE / "public_cases.json"
sys.path.insert(0, str(LAB_ROOT))

from runtime.api import ObjectiveContract, normalize_result  # noqa: E402
from runtime.hf_reference import reference_gkd  # noqa: E402
from runtime.loader import load_objective  # noqa: E402
from runtime.source_policy import admit  # noqa: E402


def make_inputs(case: dict[str, Any], device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(int(case["seed"]))
    batch = int(case["batch"])
    sequence = int(case["sequence_length"])
    student_hidden_size = int(case["student_hidden_size"])
    teacher_hidden_size = int(case["teacher_hidden_size"])
    vocab = int(case["vocab_size"])
    student_weight = (
        torch.randn(vocab, student_hidden_size, generator=generator, device=device, dtype=dtype) * 0.07
    )
    student_hidden = (
        torch.randn(batch, sequence, student_hidden_size, generator=generator, device=device, dtype=dtype)
        * 0.11
    )
    teacher_weight = (
        torch.randn(vocab, teacher_hidden_size, generator=generator, device=device, dtype=dtype) * 0.07
    )
    teacher_hidden = (
        torch.randn(batch, sequence, teacher_hidden_size, generator=generator, device=device, dtype=dtype)
        * 0.11
    )
    labels = torch.randint(vocab, (batch, sequence), generator=generator, device=device)
    ignored_count = case.get("mask_ignored_count")
    if ignored_count is None:
        labels.reshape(-1)[int(case["mask_stride"]) - 1 :: int(case["mask_stride"])] = int(case["ignore_index"])
    else:
        # A fixed stride is a pattern a candidate can special-case. Real
        # completion-only labels mask whatever the prompt happens to cover, so
        # this case ignores an irregular set of positions instead. Drawn from
        # the same seeded generator, so the case stays exactly reproducible.
        shuffled = torch.randperm(batch * sequence, generator=generator, device=labels.device)
        labels.reshape(-1)[shuffled[: int(ignored_count)]] = int(case["ignore_index"])
    # Guarantee at least one scored token per sequence regardless of stride.
    labels[:, 0] = torch.arange(batch, device=device) % vocab
    student_bias = (
        torch.randn(vocab, generator=generator, device=device, dtype=dtype) * 0.03
        if case["bias"]
        else None
    )
    teacher_bias = (
        torch.randn(vocab, generator=generator, device=device, dtype=dtype) * 0.03
        if case["bias"]
        else None
    )
    return student_weight, student_hidden, teacher_weight, teacher_hidden, labels, student_bias, teacher_bias


def clone_trainable(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Fresh leaves per execution: student side trainable, teacher side frozen."""

    student_weight, student_hidden, teacher_weight, teacher_hidden, labels, student_bias, teacher_bias = inputs
    return (
        student_weight.detach().clone().requires_grad_(True),
        student_hidden.detach().clone().requires_grad_(True),
        teacher_weight.detach().clone(),
        teacher_hidden.detach().clone(),
        labels.detach().clone(),
        None if student_bias is None else student_bias.detach().clone().requires_grad_(True),
        None if teacher_bias is None else teacher_bias.detach().clone(),
    )


def execute(objective, inputs: tuple[torch.Tensor, ...]):
    student_weight, student_hidden, teacher_weight, teacher_hidden, labels, student_bias, teacher_bias = inputs
    result = normalize_result(
        objective(
            student_weight,
            student_hidden,
            teacher_weight,
            teacher_hidden,
            labels,
            student_bias,
            teacher_bias,
        )
    )
    if not bool(torch.isfinite(result.loss)):
        raise AssertionError("non-finite loss")
    result.loss.backward()
    if teacher_weight.grad is not None or teacher_hidden.grad is not None:
        raise AssertionError("teacher tensors received gradients")
    return (
        result,
        student_weight.grad,
        student_hidden.grad,
        None if student_bias is None else student_bias.grad,
    )


def assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=1e-5, msg=label)


def cpu_correctness(
    cases: list[dict[str, Any]], requested_version: int | None
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        contract = ObjectiveContract(
            beta=float(case["beta"]),
            temperature=float(case["temperature"]),
            ignore_index=int(case["ignore_index"]),
        )
        loaded = load_objective(
            LAB_ROOT,
            contract,
            requested_version=requested_version,
        )
        inputs = make_inputs(case, torch.device("cpu"), torch.float32)
        expected = execute(
            lambda sw, sh, tw, th, y, sb, tb: reference_gkd(
                sw, sh, tw, th, y, sb, tb, contract=contract
            ),
            clone_trainable(inputs),
        )
        actual = execute(loaded.module, clone_trainable(inputs))
        assert_close(actual[0].loss, expected[0].loss, "loss")
        for index, (got, want) in enumerate(zip(actual[0].aux, expected[0].aux, strict=True)):
            assert_close(got, want, f"aux[{index}]")
        for label, got, want in zip(
            ("student_weight_grad", "student_hidden_grad", "student_bias_grad"),
            actual[1:],
            expected[1:],
            strict=True,
        ):
            if got is None or want is None:
                if got is not want:
                    raise AssertionError(f"{label} presence differs")
            else:
                assert_close(got, want, label)
        learning_rate = 3e-4
        assert_close(
            inputs[0] - learning_rate * actual[1],
            inputs[0] - learning_rate * expected[1],
            "SGD delta",
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "loss": float(actual[0].loss.detach()),
                "status": "PASS",
            }
        )
    return rows


def timed_full(objective, raw_inputs: tuple[torch.Tensor, ...], repeats: int) -> dict[str, float]:
    samples = []
    reserved = []
    allocated = []
    for index in range(repeats + 2):
        inputs = clone_trainable(raw_inputs)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        execute(objective, inputs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if index >= 2:
            samples.append(elapsed)
            reserved.append(torch.cuda.max_memory_reserved())
            allocated.append(torch.cuda.max_memory_allocated())
    return {
        "median_ms": 1000.0 * statistics.median(samples),
        "peak_reserved_mib": max(reserved) / 2**20,
        "peak_allocated_mib": max(allocated) / 2**20,
    }


def gpu_row_protocol(case: dict[str, Any]) -> dict[str, Any]:
    """Render the public, explicitly non-formal objective-row contract."""

    formal = case.get("formal", False)
    score_eligible = case.get("score_eligible", False)
    if formal is not False or score_eligible is not False:
        raise ValueError("public objective rows must remain non-formal and unscored")
    return {
        "classification": case.get(
            "classification", "PUBLIC_OBJECTIVE_DIAGNOSTIC_ONLY"
        ),
        "formal": False,
        "score_eligible": False,
        "workload_path": case.get("workload_path", "generic_objective_abi"),
        "dtype": "bfloat16",
        "execution_boundary": "objective_forward_and_backward",
        "input_contract": {
            "student_weight_shape": [int(case["vocab_size"]), int(case["student_hidden_size"])],
            "student_hidden_shape": [
                int(case["batch"]),
                int(case["sequence_length"]),
                int(case["student_hidden_size"]),
            ],
            "teacher_weight_shape": [int(case["vocab_size"]), int(case["teacher_hidden_size"])],
            "teacher_hidden_shape": [
                int(case["batch"]),
                int(case["sequence_length"]),
                int(case["teacher_hidden_size"]),
            ],
            "labels_shape": [
                int(case["batch"]),
                int(case["sequence_length"]),
            ],
            "teacher_requires_grad": False,
            "bias_present": bool(case["bias"]),
            "beta": float(case["beta"]),
            "temperature": float(case["temperature"]),
            "ignore_index": int(case["ignore_index"]),
        },
    }


def gpu_probe(
    cases: list[dict[str, Any]], repeats: int, requested_version: int | None
) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("--mode gpu requires a visible CUDA device")
    rows = []
    for case in cases:
        contract = ObjectiveContract(
            beta=float(case["beta"]),
            temperature=float(case["temperature"]),
            ignore_index=int(case["ignore_index"]),
        )
        candidate = load_objective(
            LAB_ROOT,
            contract,
            requested_version=requested_version,
        ).module.cuda()
        raw = make_inputs(case, torch.device("cuda"), torch.bfloat16)
        official_reference = (
            lambda sw, sh, tw, th, y, sb, tb: reference_gkd(
                sw, sh, tw, th, y, sb, tb, contract=contract
            )
        )
        official_reference_metrics = timed_full(official_reference, raw, repeats)
        candidate_metrics = timed_full(candidate, raw, repeats)
        rows.append(
            {
                "case_id": case["case_id"],
                "protocol": gpu_row_protocol(case),
                "official_hf_trl_reference": official_reference_metrics,
                "candidate": candidate_metrics,
                "full_objective_speedup": official_reference_metrics["median_ms"] / candidate_metrics["median_ms"],
                "reserved_ratio": candidate_metrics["peak_reserved_mib"] / official_reference_metrics["peak_reserved_mib"],
                "status": "DIAGNOSTIC_ONLY",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--candidate-version", type=int)
    args = parser.parse_args()
    if args.candidate_version is not None and not 1 <= args.candidate_version <= 999:
        parser.error("candidate version must be in [1, 999]")
    try:
        policy = admit(LAB_ROOT, requested_version=args.candidate_version)
        manifest = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        if args.mode == "cpu":
            rows = cpu_correctness(manifest["correctness"], args.candidate_version)
        else:
            rows = gpu_probe(
                manifest["gpu_objective"],
                args.repeats,
                args.candidate_version,
            )
        report = {
            "schema_version": 1,
            "status": "PASS",
            "mode": args.mode,
            "classification": (
                "PUBLIC_CPU_CORRECTNESS_QUALIFICATION"
                if args.mode == "cpu"
                else "PUBLIC_OBJECTIVE_DIAGNOSTIC_ONLY"
            ),
            "formal": False,
            "score_eligible": False,
            "source_policy": policy,
            "rows": rows,
        }
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "FAIL",
            "mode": args.mode,
            "classification": "PUBLIC_DIAGNOSTIC_ONLY",
            "formal": False,
            "score_eligible": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
