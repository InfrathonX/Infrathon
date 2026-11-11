#!/usr/bin/env python3
"""Gold-free worker that imports and executes one candidate objective."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


def make_inputs(case: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(int(case["seed"]))
    batch = int(case["batch"])
    sequence = int(case["sequence_length"])
    student_hidden_size = int(case["student_hidden_size"])
    teacher_hidden_size = int(case["teacher_hidden_size"])
    vocab = int(case["vocab_size"])
    student_weight = (
        torch.randn(vocab, student_hidden_size, generator=generator, dtype=torch.float32) * 0.07
    )
    student_hidden = (
        torch.randn(batch, sequence, student_hidden_size, generator=generator, dtype=torch.float32)
        * 0.11
    )
    teacher_weight = (
        torch.randn(vocab, teacher_hidden_size, generator=generator, dtype=torch.float32) * 0.07
    )
    teacher_hidden = (
        torch.randn(batch, sequence, teacher_hidden_size, generator=generator, dtype=torch.float32)
        * 0.11
    )
    labels = torch.randint(vocab, (batch, sequence), generator=generator)
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
    labels[:, 0] = torch.arange(batch) % vocab
    student_bias = torch.randn(vocab, generator=generator) * 0.03 if case["bias"] else None
    teacher_bias = torch.randn(vocab, generator=generator) * 0.03 if case["bias"] else None
    return student_weight, student_hidden, teacher_weight, teacher_hidden, labels, student_bias, teacher_bias


def tensor_value(value: torch.Tensor | None):
    if value is None:
        return None
    return value.detach().cpu().tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.lab_root))
    from runtime.api import ObjectiveContract, normalize_result
    from runtime.loader import load_objective

    request = json.loads(sys.stdin.readline())
    case = request["case"]
    contract = ObjectiveContract(
        beta=float(case["beta"]),
        temperature=float(case["temperature"]),
        ignore_index=int(case["ignore_index"]),
    )
    loaded = load_objective(args.lab_root, contract, int(request["version"]))
    (
        student_weight,
        student_hidden,
        teacher_weight,
        teacher_hidden,
        labels,
        student_bias,
        teacher_bias,
    ) = make_inputs(case)
    student_weight.requires_grad_(True)
    student_hidden.requires_grad_(True)
    if student_bias is not None:
        student_bias.requires_grad_(True)
    result = normalize_result(
        loaded.module(
            student_weight,
            student_hidden,
            teacher_weight,
            teacher_hidden,
            labels,
            student_bias,
            teacher_bias,
        )
    )
    result.loss.backward()
    if teacher_weight.grad is not None or teacher_hidden.grad is not None:
        raise RuntimeError("candidate objective sent gradients into teacher tensors")
    response = {
        "case_id": case["case_id"],
        "loss": tensor_value(result.loss),
        "aux": [tensor_value(item) for item in result.aux],
        "student_weight_grad": tensor_value(student_weight.grad),
        "student_hidden_grad": tensor_value(student_hidden.grad),
        "student_bias_grad": tensor_value(None if student_bias is None else student_bias.grad),
    }
    print("EDGE_RESULT " + json.dumps(response, allow_nan=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
