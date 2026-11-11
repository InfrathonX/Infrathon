#!/usr/bin/env python3
"""Gold-free worker that imports and executes one candidate objective."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


FULL_NLL_TARGET_CONTRACT = "separate_full_sequence_tensor"


def make_inputs(case: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    if case.get("separate_nll_target") is not True:
        raise ValueError("private objective cases require a separate NLL target")
    if case.get("nll_target_contract") != FULL_NLL_TARGET_CONTRACT:
        raise ValueError("private objective case has an unsupported NLL target contract")
    generator = torch.Generator(device="cpu").manual_seed(int(case["seed"]))
    total_batch = 2 * int(case["pair_batch"])
    sequence = int(case["sequence_length"])
    hidden_size = int(case["hidden_size"])
    vocab = int(case["vocab_size"])
    weight = torch.randn(vocab, hidden_size, generator=generator, dtype=torch.float32) * 0.07
    hidden = torch.randn(total_batch, sequence, hidden_size, generator=generator, dtype=torch.float32) * 0.11
    labels = torch.randint(vocab, (total_batch, sequence), generator=generator)
    labels.reshape(-1)[int(case["mask_stride"]) - 1 :: int(case["mask_stride"])] = int(case["ignore_index"])
    labels[:, 0] = torch.arange(total_batch) % vocab
    nll_target = labels.clone()
    ignored = nll_target.eq(int(case["ignore_index"]))
    replacements = torch.randint(
        vocab, (total_batch, sequence), generator=generator
    )
    nll_target = torch.where(ignored, replacements, nll_target)
    bias = torch.randn(vocab, generator=generator) * 0.03 if case["bias"] else None
    return weight, hidden, labels, bias, nll_target


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
    contract = ObjectiveContract(beta=float(case["beta"]), ignore_index=int(case["ignore_index"]))
    loaded = load_objective(args.lab_root, contract, int(request["version"]))
    weight, hidden, labels, bias, nll_target = make_inputs(case)
    weight.requires_grad_(True)
    hidden.requires_grad_(True)
    if bias is not None:
        bias.requires_grad_(True)
    result = normalize_result(loaded.module(weight, hidden, labels, bias, nll_target))
    result.loss.backward()
    response = {
        "case_id": case["case_id"],
        "loss": tensor_value(result.loss),
        "aux": [tensor_value(item) for item in result.aux],
        "weight_grad": tensor_value(weight.grad),
        "hidden_grad": tensor_value(hidden.grad),
        "bias_grad": tensor_value(None if bias is None else bias.grad),
    }
    print("EDGE_RESULT " + json.dumps(response, allow_nan=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
