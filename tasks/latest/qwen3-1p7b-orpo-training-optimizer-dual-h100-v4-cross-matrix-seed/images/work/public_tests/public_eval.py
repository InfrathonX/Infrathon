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
DEFAULT_LAB = Path("/app/workdir/orpo_lab")
SOURCE_LAB = HERE.parent / "workdir/orpo_lab"
LAB_ROOT = Path(os.environ.get("ORPO_LAB_ROOT", DEFAULT_LAB if DEFAULT_LAB.is_dir() else SOURCE_LAB))
CASES_PATH = HERE / "public_cases.json"
sys.path.insert(0, str(LAB_ROOT))

from runtime.api import ObjectiveContract, normalize_result  # noqa: E402
from runtime.hf_reference import reference_orpo  # noqa: E402
from runtime.loader import load_objective  # noqa: E402
from runtime.source_policy import admit  # noqa: E402


FULL_NLL_TARGET_CONTRACT = "separate_full_sequence_tensor"


def make_inputs(case: dict[str, Any], device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    if case.get("separate_nll_target") is not True:
        raise ValueError("every objective case requires a separate full-sequence NLL target")
    if case.get("nll_target_contract") != FULL_NLL_TARGET_CONTRACT:
        raise ValueError("objective case has an unsupported NLL target contract")
    generator = torch.Generator(device=device).manual_seed(int(case["seed"]))
    total_batch = 2 * int(case["pair_batch"])
    sequence = int(case["sequence_length"])
    hidden_size = int(case["hidden_size"])
    vocab = int(case["vocab_size"])
    weight = torch.randn(vocab, hidden_size, generator=generator, device=device, dtype=dtype) * 0.07
    hidden = torch.randn(total_batch, sequence, hidden_size, generator=generator, device=device, dtype=dtype) * 0.11
    labels = torch.randint(vocab, (total_batch, sequence), generator=generator, device=device)
    labels.reshape(-1)[int(case["mask_stride"]) - 1 :: int(case["mask_stride"])] = int(case["ignore_index"])
    labels[:, 0] = torch.arange(total_batch, device=device) % vocab
    # OR labels intentionally contain ignored positions.  The chosen NLL target
    # is a distinct, complete sequence target: no token is masked out.  This
    # mirrors official TRL ORPO semantics, where the odds-ratio term is scored
    # on completion labels while chosen NLL covers prompt plus completion.
    nll_target = labels.clone()
    ignored = nll_target.eq(int(case["ignore_index"]))
    replacements = torch.randint(
        vocab,
        (total_batch, sequence),
        generator=generator,
        device=device,
    )
    nll_target = torch.where(ignored, replacements, nll_target)
    bias = (
        torch.randn(vocab, generator=generator, device=device, dtype=dtype) * 0.03
        if case["bias"]
        else None
    )
    return weight, hidden, labels, bias, nll_target


def clone_trainable(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    weight, hidden, labels, bias, nll_target = inputs
    if nll_target is None:
        raise ValueError("objective inputs require a separate full-sequence NLL target")
    return (
        weight.detach().clone().requires_grad_(True),
        hidden.detach().clone().requires_grad_(True),
        labels.detach().clone(),
        None if bias is None else bias.detach().clone().requires_grad_(True),
        nll_target.detach().clone(),
    )


def execute(objective, inputs: tuple[torch.Tensor, ...]):
    weight, hidden, labels, bias, nll_target = inputs
    result = normalize_result(objective(weight, hidden, labels, bias, nll_target))
    if not bool(torch.isfinite(result.loss)):
        raise AssertionError("non-finite loss")
    result.loss.backward()
    return result, weight.grad, hidden.grad, None if bias is None else bias.grad


def assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=1e-5, msg=label)


def cpu_correctness(
    cases: list[dict[str, Any]], requested_version: int | None
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        contract = ObjectiveContract(beta=float(case["beta"]), ignore_index=int(case["ignore_index"]))
        loaded = load_objective(
            LAB_ROOT,
            contract,
            requested_version=requested_version,
        )
        inputs = make_inputs(case, torch.device("cpu"), torch.float32)
        expected = execute(
            lambda w, h, y, b, n: reference_orpo(w, h, y, b, n, contract=contract),
            clone_trainable(inputs),
        )
        actual = execute(loaded.module, clone_trainable(inputs))
        assert_close(actual[0].loss, expected[0].loss, "loss")
        for index, (got, want) in enumerate(zip(actual[0].aux, expected[0].aux, strict=True)):
            assert_close(got, want, f"aux[{index}]")
        for label, got, want in zip(
            ("weight_grad", "hidden_grad", "bias_grad"), actual[1:], expected[1:], strict=True
        ):
            if got is None or want is None:
                if got is not want:
                    raise AssertionError(f"{label} presence differs")
            else:
                assert_close(got, want, label)
        learning_rate = 3e-4
        assert_close(inputs[0] - learning_rate * actual[1], inputs[0] - learning_rate * expected[1], "SGD delta")
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
    if case.get("separate_nll_target") is not True:
        raise ValueError("public objective rows require a separate NLL target")
    if case.get("nll_target_contract") != FULL_NLL_TARGET_CONTRACT:
        raise ValueError("public objective row has an unsupported NLL target contract")
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
            "weight_shape": [int(case["vocab_size"]), int(case["hidden_size"])],
            "hidden_states_shape": [
                2 * int(case["pair_batch"]),
                int(case["sequence_length"]),
                int(case["hidden_size"]),
            ],
            "labels_shape": [
                2 * int(case["pair_batch"]),
                int(case["sequence_length"]),
            ],
            "chosen_first": True,
            "bias_present": bool(case["bias"]),
            "nll_target": FULL_NLL_TARGET_CONTRACT,
            "beta": float(case["beta"]),
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
        contract = ObjectiveContract(beta=float(case["beta"]), ignore_index=int(case["ignore_index"]))
        candidate = load_objective(
            LAB_ROOT,
            contract,
            requested_version=requested_version,
        ).module.cuda()
        raw = make_inputs(case, torch.device("cuda"), torch.bfloat16)
        official_reference = (
            lambda w, h, y, b, n: reference_orpo(
                w, h, y, b, n, contract=contract
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
